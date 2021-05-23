import os
from PIL import Image, ImageDraw

from src.utils.logger import Logger
Logger.setLevel(Logger.Level.INFO)

from src.utils import common
from src.utils.detection import DetectionRawOutput, BBox

BBOX_MIN_IOU = 0.5
RETURN_WHEN_CRITICAL = False

def draw_detections_and_show(img_name, detections, labels, color='green'):
    image = Image.open(common.get_input_file_from_name(img_name)).convert('RGB')
    draw = ImageDraw.Draw(image)
    for obj in detections:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                    outline=color)
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                fill=color)
    image.show()

def get_golden_for_sdc_file(sdc_file):
    model_name, image_name, timestamp = common.parse_sdc_out_filename(sdc_file)
    return common.get_dft_golden_filename(model_name, image_name)

def validate_model_and_input_sizes(gold_data, sdc_data):
    g_input_image_scale = gold_data.get('input_image_scale')
    g_model_input_size = gold_data.get('model_input_size')

    sdc_input_image_scale = sdc_data.get('input_image_scale')
    sdc_model_input_size = sdc_data.get('model_input_size')

    assert g_input_image_scale == sdc_input_image_scale, "Input image sizes mismatch"
    assert g_model_input_size == sdc_model_input_size, "Model input sizes mismatch"

    return g_model_input_size, g_input_image_scale

def is_sdc_critical(sdc_file, labels, threshold=0) -> tuple:
    critical = False
    errors = []

    gold_file = get_golden_for_sdc_file(sdc_file)

    try:
        gold_data = common.load_tensors_from_file(gold_file)
        sdc_data = common.load_tensors_from_file(sdc_file)
    except:
        errors.append(f"Corrupted file")
        return False, errors

    input_size, img_scale = validate_model_and_input_sizes(gold_data, sdc_data)

    golden_dets = DetectionRawOutput.objs_from_data(gold_data, threshold)
    sdc_dets = DetectionRawOutput.objs_from_data(sdc_data, threshold)

    # Compare amount of detections
    if len(golden_dets) != len(sdc_dets):
        errors.append(f"Number of detections mismatch (expected: {len(golden_dets)}, got: {len(sdc_dets)})")
        if RETURN_WHEN_CRITICAL: return True, errors
        else: critical = True

    # Compare each object
    for i in range(min(len(golden_dets), len(sdc_dets))):
        gold_det = golden_dets[i]
        sdc_det = sdc_dets[i]

        if sdc_det.id != gold_det.id:
            errors.append(f"Object {i}: Wrong class (expected: {labels.get(gold_det.id)}, got: {labels.get(sdc_det.id)})")
            if RETURN_WHEN_CRITICAL: return True, errors
            else: critical = True

        if sdc_det.score != gold_det.score:
            errors.append(f"Object {i}: Scores mismatch (expected: {gold_det.score}, got: {sdc_det.score})")
        
        iou = BBox.iou(sdc_det.bbox, gold_det.bbox)
        if iou < BBOX_MIN_IOU:
            errors.append(f"Object {i}: BBox IoU < {BBOX_MIN_IOU} (IoU: {iou})")
            if RETURN_WHEN_CRITICAL: return True, errors
            else: critical = True
    
    # draw_detections_and_show(img_name, golden_dets, labels)
    # draw_detections_and_show(img_name, sdc_dets, labels, color='red')

    return critical, errors

def main():
    thresh = 0.5
    labels_file = 'labels/coco_labels.txt'

    labels = common.read_label_file(labels_file)

    total = 0
    criticals = 0

    for sdc_filename in os.listdir(common.OUTPUTS_DIR):
        if sdc_filename.startswith('sdc'):
            total += 1

            sdc_file = f"{common.OUTPUTS_DIR}/{sdc_filename}"
            critical, errors = is_sdc_critical(sdc_file, labels, thresh)
            print(f"{sdc_filename}: {'CRITICAL' if critical else 'OK'}")
            if critical:
                print("\n".join(map(lambda e: "\t* " + e, errors)))
                criticals += 1

    print("\n---- SUMMARY ----")
    print("Total: {:d}".format(total))
    print("Criticals: {:d} ({:.2f} %)".format(criticals, criticals*100/total))

if __name__ == "__main__":
    main()