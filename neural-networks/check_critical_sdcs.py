#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
from typing import List

from src.utils.logger import Logger
Logger.setLevel(Logger.Level.INFO)

from src.utils import common
from src.utils.common import ModelTask, ModelsManager
from src.utils.detection import DetectionRawOutput, BBox, match_detections
from src.utils.classification import get_classes_from_scores

# Configs
# - Common
RETURN_WHEN_CRITICAL = False
# - Detection
MATCH_BBOX_IOU = True
BBOX_MIN_IOU = 0.5
# - Classification
TOP_K_CLASSES = 3

class SDCOutputException(Exception): 
    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {super().__str__()}"

class InterruptAnalysisException(SDCOutputException): pass
class NotSDCOutputFileException(SDCOutputException): pass
class CorruptedFileException(SDCOutputException): pass
class ImageScaleMismatchException(SDCOutputException): pass
class InputSizeMismatchException(SDCOutputException): pass
class GoldenFileNotFoundException(SDCOutputException): pass

class PredictionError:
    def __init__(self, msg: str, is_critical = False) -> None:
        self.is_critical = is_critical
        self.msg = msg
    
    @property
    def type(self):
        return self.__class__.__name__

class WrongDetsNumberPredictionError(PredictionError):
    def __init__(self, expected: int, got: int) -> None:
        super().__init__(f"Wrong number of detections (expected: {expected}, got: {got})", is_critical=True)

class BBoxPredictionError(PredictionError):
    def __init__(self, iou: float) -> None:
        assert iou < BBOX_MIN_IOU
        super().__init__(f"BBox IoU < {BBOX_MIN_IOU} (IoU: {iou})", is_critical=True)

class WrongClassPredictionError(PredictionError):
    def __init__(self, expected: str, got: str, iou: float=None, is_critical=True) -> None:
        if iou:
            super().__init__(f"Wrong class (expected: {expected}, got: {got}, IoU: {iou})", is_critical=iou >= MATCH_BBOX_IOU)
        else:
            super().__init__(f"Wrong class (expected: {expected}, got: {got})", is_critical=is_critical)

class WrongScorePredictionError(PredictionError):
    def __init__(self, expected: float, got: float) -> None:
        super().__init__(f"Scores mismatch (expected: {expected}, got: {got})", is_critical=False)

class ModelSDCsStats():
    def __init__(self) -> None:
        # General stats
        self.total_sdc_outputs = 0
        self.critical_sdc_outputs = 0
        self.ignored_sdc_outputs = 0

        # Prediction errors stats
        self.total_prediction_errors = 0
        self.prediction_errors_by_type = { err_class.__name__: 0 for err_class in PredictionError.__subclasses__() }

class ModelsSDCsStatsManager():
    MODELS_SDCs_STATS_MAP = { m.name: ModelSDCsStats() for m in ModelsManager.MODELS_LIST }

    @staticmethod
    def get_stats_by_model_name(model_name: str):
        return ModelsSDCsStatsManager.MODELS_SDCs_STATS_MAP.get(model_name)

class SDCOutput:
    SCORE_THRESHOLD = 0.5

    @staticmethod
    def set_score_threshold(thresh: float):
        SDCOutput.SCORE_THRESHOLD = thresh

    def __init__(self, file: str) -> None:
        if not Path(file).stem.startswith('sdc'):
            raise NotSDCOutputFileException(f"{file} is not a SDC output file")

        self.file: str = file

        self.model_name: str = None
        self.image_name: str = None
        self.timestamp: int = None
        self.gold_file: str = None

        self.prediction_errors: List[PredictionError] = []
        self.is_critical: bool = False
        self.task: ModelTask = None

        self.analyze()

    def __str__(self) -> str:
        lines = [f"{os.path.basename(self.file)}: {'CRITICAL' if self.is_critical else 'OK'}"]
        for err in self.prediction_errors:
            lines.append(f"\t* {err.msg}" )
        return '\n'.join(lines)

    def print(self, file=None):
        if file:
            print(self, file=file)
        else:
            print(self)

    def prepare_for_analysis(self):
        self.model_name, self.image_name, self.timestamp = common.parse_sdc_out_filename(self.file)
        self.gold_file = common.get_golden_filename(self.model_name, self.image_name)

        try:
            gold_data = common.load_tensors_from_file(self.gold_file)
            sdc_data = common.load_tensors_from_file(self.file)
        except FileNotFoundError as ex:
            raise GoldenFileNotFoundException(ex)
        except Exception as ex:
            raise CorruptedFileException(ex)

        self.task = ModelsManager.get_by_name(self.model_name).task
        labels = ModelsManager.get_model_labels(self.model_name)

        return gold_data, sdc_data, labels

    def analyze(self):
        gold_data, sdc_data, labels = self.prepare_for_analysis()
        
        try:
            if self.task == ModelTask.Detection:
                self.analyze_detection(gold_data, sdc_data, labels)
            elif self.task == ModelTask.Classification:
                self.analyze_classification(gold_data, sdc_data, labels)
            else:
                raise SDCOutputException("Unsupported model task")
        except InterruptAnalysisException:
            # Stop analysis on first critical error
            pass        

    def add_prediction_error(self, error: PredictionError):
        self.prediction_errors.append(error)
        if error.is_critical:
            self.is_critical = True
            if RETURN_WHEN_CRITICAL:
                # Stop analysis on first critical error
                raise InterruptAnalysisException()

    # Detection

    def validate_model_and_input_sizes(self, gold_data, sdc_data):
        g_input_image_scale = gold_data.get('input_image_scale')
        g_model_input_size = gold_data.get('model_input_size')

        sdc_input_image_scale = sdc_data.get('input_image_scale')
        sdc_model_input_size = sdc_data.get('model_input_size')

        if g_input_image_scale != sdc_input_image_scale:
            raise ImageScaleMismatchException(f"Expected: {g_input_image_scale}, got: {sdc_input_image_scale}")

        if g_model_input_size != sdc_model_input_size:
            raise InputSizeMismatchException(f"Expected: {g_model_input_size}, got: {sdc_model_input_size}")

        return g_model_input_size, g_input_image_scale
        
    def analyze_detection(self, gold_data, sdc_data, labels):
        self.validate_model_and_input_sizes(gold_data, sdc_data)

        gold_dets = DetectionRawOutput.objs_from_data(gold_data, SDCOutput.SCORE_THRESHOLD)
        sdc_dets = DetectionRawOutput.objs_from_data(sdc_data, SDCOutput.SCORE_THRESHOLD)

        # Compare amount of detections
        n_gold_dets = len(gold_dets)
        n_sdc_dets = len(sdc_dets)
        if n_gold_dets != n_sdc_dets:
            self.add_prediction_error(WrongDetsNumberPredictionError(expected=n_gold_dets, got=n_sdc_dets))

        det_pairs = match_detections(gold_dets, sdc_dets) if MATCH_BBOX_IOU else \
                        [(gold_dets[i], sdc_dets[i], BBox.iou(gold_dets[i].bbox, sdc_dets[i].bbox)) for i in range(min(n_gold_dets, len(n_sdc_dets)))]
        
        # Compare each detection pair
        for (gold_det, sdc_det, iou) in det_pairs:
            if sdc_det.id != gold_det.id:
                sdc_det_class = labels.get(sdc_det.id)
                gold_det_class = labels.get(gold_det.id)
                self.add_prediction_error(WrongClassPredictionError(gold_det_class, sdc_det_class, iou=iou))

            if iou < BBOX_MIN_IOU:
                self.add_prediction_error(BBoxPredictionError(iou))

            if sdc_det.score != gold_det.score:
                self.add_prediction_error(WrongScorePredictionError(gold_det.score, sdc_det.score))

            # draw_detections_and_show(self.image_name, gold_dets, labels)
            # draw_detections_and_show(self.image_name, sdc_dets, labels, color='red')

    # Classification

    def analyze_classification(self, gold_data, sdc_data, labels):
        gold_scores = gold_data.get('scores')
        sdc_scores = sdc_data.get('scores')

        top_gold_classes = get_classes_from_scores(gold_scores, TOP_K_CLASSES, SDCOutput.SCORE_THRESHOLD)
        top_sdc_classes = get_classes_from_scores(sdc_scores, TOP_K_CLASSES, SDCOutput.SCORE_THRESHOLD)

        for i, (gold_class, sdc_class) in enumerate(zip(top_gold_classes, top_sdc_classes)):
            if gold_class.id != sdc_class.id:
                gold_class_label = labels.get(gold_class.id)
                sdc_class_label = labels.get(sdc_class.id)
                critical = i == 0 # Critical only if top 1 class
                self.add_prediction_error(WrongClassPredictionError(gold_class_label, sdc_class_label, is_critical=critical))

            if gold_class.score != sdc_class.score:
                self.add_prediction_error(WrongScorePredictionError(gold_class.score, sdc_class.score))

def print_stdout_and_file(string, file, indent_level=0):
    content = '  ' * indent_level + string
    print(content)
    print(content, file=file)

def percent(n, d):
    return n*100/d if n != 0 else 0.0

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('sdcs_dir', help='Path to SDC output files directory')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Threshold for detection/classification score')
    args = parser.parse_args()

    thresh = args.threshold
    sdcs_dir = args.sdcs_dir.rstrip('/')

    # Generate output file names
    sdcs_sub_dir = Path(sdcs_dir).stem
    out_file_suffix = f"-{sdcs_sub_dir}.txt" if sdcs_sub_dir != "outputs" else ".txt"
    analysis_full_out_file = os.path.join(common.OUTPUTS_DIR, "sdcs-analysis-full" + out_file_suffix)
    analysis_summary_out_file = os.path.join(common.OUTPUTS_DIR, "sdcs-analysis-summary" + out_file_suffix)

    SDCOutput.set_score_threshold(thresh)

    with open(analysis_full_out_file, 'w') as full_out_file:
        for path, dirs, files in os.walk(sdcs_dir):
            Logger.info(f"Analyzing {len(files)} files in `{path}`")
            for filename in files:
                try:
                    sdc_file = os.path.join(path, filename)

                    sdc = SDCOutput(sdc_file)
                    sdc.print(file=full_out_file)

                    model_stats = ModelsSDCsStatsManager.get_stats_by_model_name(sdc.model_name)

                    model_stats.total_sdc_outputs += 1
                    if sdc.is_critical:
                        model_stats.critical_sdc_outputs += 1
                        model_stats.total_prediction_errors += len(sdc.prediction_errors)
                        for err in sdc.prediction_errors:
                            model_stats.prediction_errors_by_type[err.type] += 1

                except NotSDCOutputFileException:
                    Logger.info(f"Ignoring not SDC file: {filename}")
                except SDCOutputException as ex:
                    Logger.info(f"Ignoring SDC file: {filename} - {ex}")
                    if model_stats:
                        model_stats.ignored_sdc_outputs += 1
                except Exception as ex:
                    raise ex
            
        full_out_file.close()

    with open(analysis_summary_out_file, 'w') as summary_out_file:
        print_stdout_and_file("\n---- SDC OUTPUT ANALYSIS SUMMARY ----", summary_out_file)
        print_stdout_and_file("", summary_out_file)

        print_stdout_and_file(" - Full analysis output file: {:s}".format(analysis_full_out_file), summary_out_file)
        print_stdout_and_file(" - Threshold: {:.2f}".format(SDCOutput.SCORE_THRESHOLD), summary_out_file)
        print_stdout_and_file(" * `Total SDC outputs` accounts for the number of executions in which at least one output value," \
            "associated to a prediction with score above 0.3 in the golden execution, was corrupted.", summary_out_file)
        print_stdout_and_file("    See `DETECTION_THRESHOLD` in `run_detection.py` and `CLASSIFICATION_THRESHOLD` in `run_classification.py`.", summary_out_file)
        print_stdout_and_file("", summary_out_file)

        print_stdout_and_file(">  MODELS", summary_out_file)
        print_stdout_and_file("", summary_out_file)

        for model_name, stats in ModelsSDCsStatsManager.MODELS_SDCs_STATS_MAP.items():
            print_stdout_and_file(f"{model_name}:", summary_out_file, indent_level=1)

            model_task = ModelsManager.get_by_name(model_name).task
            total = stats.total_sdc_outputs
            criticals = stats.critical_sdc_outputs
            ignored = stats.ignored_sdc_outputs
            total_pred_errs = stats.total_prediction_errors

            print_stdout_and_file("- Task: {:s}".format(model_task.value), summary_out_file, indent_level=2)
            print_stdout_and_file("- Total SDC outputs*: {:d}".format(total), summary_out_file, indent_level=2)
            print_stdout_and_file("- Critical SDC outputs: {:d} ({:.2f} %)".format(criticals, percent(criticals, total)), summary_out_file, indent_level=2)
            print_stdout_and_file("- Ignored SDC outputs: {:d} ({:.2f} %)".format(ignored, percent(ignored, total)), summary_out_file, indent_level=2)
            print_stdout_and_file("- Total prediction errors: {:d}".format(total_pred_errs), summary_out_file, indent_level=2)
            for err_type, err_count in sorted(stats.prediction_errors_by_type.items(), key=lambda item: item[1], reverse=True):
                err_name = err_type.rstrip('PredictionError')
                print_stdout_and_file("- {:s}: {:d} ({:.2f} %)".format(err_name, err_count, percent(err_count, total_pred_errs)), summary_out_file, indent_level=3)
            print_stdout_and_file("", summary_out_file)

        summary_out_file.close()

if __name__ == "__main__":
    main()