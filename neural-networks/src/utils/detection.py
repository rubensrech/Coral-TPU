# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
import numpy as np

from src.utils import common

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])
"""Represents a detected object.
  .. py:attribute:: id
      The object's class id.
  .. py:attribute:: score
      The object's prediction score.
  .. py:attribute:: bbox
      A :obj:`BBox` object defining the object's location.
"""


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """The bounding box for a detected object.
    .. py:attribute:: xmin
        X-axis start point
    .. py:attribute:: ymin
        Y-axis start point
    .. py:attribute:: xmax
        X-axis end point
    .. py:attribute:: ymax
        Y-axis end point
    """
    __slots__ = ()

    @property
    def width(self):
        """The bounding box width."""
        return self.xmax - self.xmin

    @property
    def height(self):
        """The bounding box height."""
        return self.ymax - self.ymin

    @property
    def area(self):
        """The bound box area."""
        return self.width * self.height

    @property
    def valid(self):
        """Indicates whether bounding box is valid or not (boolean).
        A valid bounding box has xmin <= xmax and ymin <= ymax (equivalent
        to width >= 0 and height >= 0).
        """
        return self.width >= 0 and self.height >= 0

    def scale(self, sx, sy):
        """Scales the bounding box.
        Args:
          sx (float): Scale factor for the x-axis.
          sy (float): Scale factor for the y-axis.
        Returns:
          A :obj:`BBox` object with the rescaled dimensions.
        """
        return BBox(xmin=sx * self.xmin,
                    ymin=sy * self.ymin,
                    xmax=sx * self.xmax,
                    ymax=sy * self.ymax)

    def translate(self, dx, dy):
        """Translates the bounding box position.
        Args:
          dx (int): Number of pixels to move the box on the x-axis.
          dy (int): Number of pixels to move the box on the y-axis.
        Returns:
          A :obj:`BBox` object at the new position.
        """
        return BBox(xmin=dx + self.xmin,
                    ymin=dy + self.ymin,
                    xmax=dx + self.xmax,
                    ymax=dy + self.ymax)

    def map(self, f):
        """Maps all box coordinates to a new position using a given function.
        Args:
          f: A function that takes a single coordinate and returns a new one.
        Returns:
          A :obj:`BBox` with the new coordinates.
        """
        return BBox(xmin=f(self.xmin),
                    ymin=f(self.ymin),
                    xmax=f(self.xmax),
                    ymax=f(self.ymax))

    @staticmethod
    def intersect(a, b):
        """Gets a box representing the intersection between two boxes.
        Args:
          a: :obj:`BBox` A.
          b: :obj:`BBox` B.
        Returns:
          A :obj:`BBox` representing the area where the two boxes intersect
          (may be an invalid box, check with :func:`valid`).
        """
        return BBox(xmin=max(a.xmin, b.xmin),
                    ymin=max(a.ymin, b.ymin),
                    xmax=min(a.xmax, b.xmax),
                    ymax=min(a.ymax, b.ymax))

    @staticmethod
    def union(a, b):
        """Gets a box representing the union of two boxes.
        Args:
          a: :obj:`BBox` A.
          b: :obj:`BBox` B.
        Returns:
          A :obj:`BBox` representing the unified area of the two boxes
          (always a valid box).
        """
        return BBox(xmin=min(a.xmin, b.xmin),
                    ymin=min(a.ymin, b.ymin),
                    xmax=max(a.xmax, b.xmax),
                    ymax=max(a.ymax, b.ymax))

    @staticmethod
    def iou(a, b):
        """Gets the intersection-over-union value for two boxes.
        Args:
          a: :obj:`BBox` A.
          b: :obj:`BBox` B.
        Returns:
          The intersection-over-union value: 1.0 meaning the two boxes are
          perfectly aligned, 0 if not overlapping at all (invalid intersection).
        """
        intersection = BBox.intersect(a, b)
        if not intersection.valid:
            return 0.0
        area = intersection.area
        return area / (a.area + b.area - area)

class DetectionRawOutput(collections.namedtuple('DetectionRawOutput', ['boxes', 'class_ids', 'scores', 'count'])):
    """Represents the raw output tensors of the interpreter.
        .. py:attribute:: boxes
            Array containing raw values for all boxes that outcome from the detection
        .. py:attribute:: class_ids
            Array containing raw values for all class ids that outcome from the detection
        .. py:attribute:: scores
            Array containing raw values for all scores that outcome from the detection
        .. py:attribute:: count
            Integer value representing the number of objects that outcome from the detection
    """
    __slots__ = ()

    def save_to_file(self, filename):
        np.save(filename, self._asdict())
    
    @staticmethod
    def from_file(filename):
        data = np.load(filename, allow_pickle=True).item()
        return DetectionRawOutput(
            boxes=data['boxes'],
            class_ids=data['class_ids'],
            scores=data['scores'],
            count=data['count'])

    def get_objects(self, input_size, img_scale=(1.,1.), threshold=-float('inf')):
        width, height = input_size
        img_scale_x, img_scale_y = img_scale
        sx, sy = width / img_scale_x, height / img_scale_y

        def make_object(i):
            ymin, xmin, ymax, xmax = self.boxes[i]
            return Object(
                id=self.class_ids[i],
                score=self.scores[i],
                bbox=BBox(xmin, ymin, xmax, ymax).scale(sx, sy).map(int))

        return [make_object(i) for i in range(self.count) if self.scores[i] >= threshold]
    

def get_raw_output(interpreter):
    return DetectionRawOutput(
        boxes=common.output_tensor(interpreter, 0)[0],
        class_ids=common.output_tensor(interpreter, 1)[0],
        scores=common.output_tensor(interpreter, 2)[0],
        count=int(common.output_tensor(interpreter, 3)[0]))

def get_objects(interpreter, img_scale=(1., 1.), threshold=-float('inf')):
    return get_raw_output(interpreter).get_objects(common.input_size(interpreter), img_scale)