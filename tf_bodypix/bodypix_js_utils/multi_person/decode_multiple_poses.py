# based on;
# https://github.com/tensorflow/tfjs-models/blob/body-pix-v2.0.5/body-pix/src/multi_person/decode_multiple_poses.ts

import logging

from typing import Dict, List

from tf_bodypix.bodypix_js_utils.types import (
    Pose, TensorBuffer3D, Vector2D,
    Keypoint
)
from tf_bodypix.bodypix_js_utils.build_part_with_score_queue import (
    build_part_with_score_queue
)

from .util import getImageCoords, squared_distance_vector
from .decode_pose import decodePose


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


kLocalMaximumRadius = 1


def withinNmsRadiusOfCorrespondingPoint(
    poses: List[Pose],
    squaredNmsRadius: float,
    vector: Vector2D,
    keypointId: int
) -> bool:
    return any(
        squared_distance_vector(
            vector, pose.keypoints[keypointId].position
        ) <= squaredNmsRadius
        for pose in poses
    )


def getInstanceScore(
    existingPoses: List[Pose],
    squaredNmsRadius: float,
    instanceKeypoints: Dict[int, Keypoint]
) -> float:
    LOGGER.debug('instanceKeypoints: %s', instanceKeypoints)
    notOverlappedKeypointScores = sum((
        keypoint.score
        for keypointId, keypoint in instanceKeypoints.items()
        if not withinNmsRadiusOfCorrespondingPoint(
            existingPoses, squaredNmsRadius,
            keypoint.position, keypointId
        )
    ))

    return notOverlappedKeypointScores / len(instanceKeypoints)


def decodeMultiplePoses(
    scoresBuffer: TensorBuffer3D,
    offsetsBuffer: TensorBuffer3D,
    displacementsFwdBuffer: TensorBuffer3D,
    displacementsBwdBuffer: TensorBuffer3D,
    outputStride: int,
    maxPoseDetections: int,
    scoreThreshold: float = 0.5,
    nmsRadius: float = 20
) -> List[Pose]:
    poses: List[Pose] = []

    # 結局 scoreThreshold でキーポイントをフィルタしてるだけ
    # 複数人検出されることがあるので、キーポイントIDごとに最大検出人数分の値を抽出している
    # 処理が複雑になるので無理に複数人検出を実装する必要はない
    queue = build_part_with_score_queue(
        scoreThreshold, # scoreThreshold = 0.5
        kLocalMaximumRadius, # kLocalMaximumRadius = 1
        scoresBuffer # self.heatmap_logits
    )
    # LOGGER.debug('queue: %s', queue)

    squaredNmsRadius = nmsRadius * nmsRadius # 40

    # Generate at most maxDetections object instances per image in
    # decreasing root part score order.
    while len(poses) < maxPoseDetections and queue:
        # The top element in the queue is the next root candidate.
        root = queue.popleft()

        # Part-based non-maximum suppression: We reject a root candidate if it
        # is within a disk of `nmsRadius` pixels from the corresponding part of
        # a previously detected instance.
        """
        self.heatmap_logits.shape = scoresBuffer = [1, 14, 21, 17]
        self.short_offsets.shape = offsetsBuffer = [1, 14, 21, 34]
        self.displacement_fwd.shape = displacementsFwdBuffer = [1, 14, 21, 32]
        self.displacement_bwd.shape = displacementsBwdBuffer = [1, 14, 21, 32]
        output_stride = 16
        maxPoseDetections = 2

        rootImageCoords: 各yx座標に対して strides=16 を掛け算したあとに offset を加算したもの -> 関連性の高いキーポイントを距離 (radius) を基に検出するための距離計算にしか使ってない
        """
        rootImageCoords = \
            getImageCoords(
                part=root.part,
                outputStride=outputStride,
                offsets=offsetsBuffer
            )
        if withinNmsRadiusOfCorrespondingPoint(
            poses, squaredNmsRadius, rootImageCoords, root.part.keypoint_id
        ):
            continue

        # Start a new detection instance at the position of the root.
        """
        self.heatmap_logits.shape = scoresBuffer = [1, 14, 21, 17]
        self.short_offsets.shape = offsetsBuffer = [1, 14, 21, 34]
        self.displacement_fwd.shape = displacementsFwdBuffer = [1, 14, 21, 32]
        self.displacement_bwd.shape = displacementsBwdBuffer = [1, 14, 21, 32]
        output_stride = 16
        """
        keypoints = \
            decodePose(
                root=root,
                scores=scoresBuffer,
                offsets=offsetsBuffer,
                outputStride=outputStride,
                displacementsFwd=displacementsFwdBuffer,
                displacementsBwd=displacementsBwdBuffer,
            )

        # LOGGER.debug('keypoints: %s', keypoints)

        score = getInstanceScore(poses, squaredNmsRadius, keypoints)

        poses.append(Pose(keypoints=keypoints, score=score))

    return poses
