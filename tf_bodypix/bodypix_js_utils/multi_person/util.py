# based on;
# https://github.com/tensorflow/tfjs-models/blob/body-pix-v2.0.5/body-pix/src/multi_person/util.ts

import logging

from ..types import Part, TensorBuffer3D, Vector2D
from ..keypoints import NUM_KEYPOINTS


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def getOffsetPoint(
    y: float, x: float, keypoint_id: int, offsets: TensorBuffer3D
) -> Vector2D:
    """
    self.short_offsets.shape = offsetsBuffer = offsets = [1, 14, 21, 34]
    NUM_KEYPOINTS = 17
    つまり、offsetsには [yyyyyyyyyyyyyyyyyxxxxxxxxxxxxxxxxx] が入っている
    """
    return Vector2D(
        y=offsets[int(y), int(x), keypoint_id],
        x=offsets[int(y), int(x), keypoint_id + NUM_KEYPOINTS]
    )


def getImageCoords(
    part: Part,
    outputStride: int,
    offsets: TensorBuffer3D
) -> Vector2D:
    LOGGER.debug('part: %s', part)
    """
    self.heatmap_logits.shape = scoresBuffer = [1, 14, 21, 17]
    self.short_offsets.shape = offsetsBuffer = offsets = [1, 14, 21, 34]
    self.displacement_fwd.shape = displacementsFwdBuffer = [1, 14, 21, 32]
    self.displacement_bwd.shape = displacementsBwdBuffer = [1, 14, 21, 32]
    output_stride = 16
    maxPoseDetections = 2

    self.short_offsets.shape = offsetsBuffer = offsets = [1, 14, 21, 34]
    NUM_KEYPOINTS = 17
    つまり、offsetsには [yyyyyyyyyyyyyyyyyxxxxxxxxxxxxxxxxx] が入っている

    offset_point は [y,x] 座標点ごとに加算すべき offset 値
    """
    offset_point = getOffsetPoint(
        y=part.heatmap_y,
        x=part.heatmap_x,
        keypoint_id=part.keypoint_id,
        offsets=offsets
    )
    LOGGER.debug('offset_point: %s', offset_point)
    LOGGER.debug('offsets.shape: %s', offsets.shape)
    return Vector2D(
        x=part.heatmap_x * outputStride + offset_point.x,
        y=part.heatmap_y * outputStride + offset_point.y
    )


def clamp(a: int, min_value: int, max_value: int) -> int:
    return min(max_value, max(min_value, a))


def squaredDistance(
    y1: float, x1: float, y2: float, x2: float
) -> float:
    dy = y2 - y1
    dx = x2 - x1
    return dy * dy + dx * dx


def squared_distance_vector(a: Vector2D, b: Vector2D) -> float:
    return squaredDistance(a.y, a.x, b.y, b.x)


def addVectors(a: Vector2D, b: Vector2D) -> Vector2D:
    return Vector2D(x=a.x + b.x, y=a.y + b.y)
