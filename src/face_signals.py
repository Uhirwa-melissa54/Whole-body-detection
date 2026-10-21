# src/face_signals.py

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)

MOUTH_LEFT, MOUTH_RIGHT = 61, 291
LIP_TOP, LIP_BOTTOM = 13, 14

FACE_LEFT, FACE_RIGHT = 234, 454


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(
    points: np.ndarray,
    idx: Tuple[int, ...],
) -> float:
    p1, p2, p3, p4, p5, p6 = (points[i] for i in idx)

    width = max(distance(p1, p4), 1e-6)

    return (distance(p2, p6) + distance(p3, p5)) / (2.0 * width)


@dataclass
class FaceSignals:
    ear: float
    blink: bool
    eyes_closed: bool
    smile_score: float
    smiling: bool


class FaceSignalExtractor:
    def __init__(
        self,
        ear_threshold: float = 0.21,
        blink_min_frames: int = 2,
        blink_max_frames: int = 7,
        closed_frames: int = 8,
        smile_on: float = 0.055,
        smile_off: float = 0.028,
    ):
        self.ear_threshold = ear_threshold
        self.blink_min_frames = blink_min_frames
        self.blink_max_frames = blink_max_frames
        self.closed_frames = closed_frames
        self.smile_on = smile_on
        self.smile_off = smile_off

        self.low_ear_frames = 0
        self.smiling = False
        self.cooldown = 0
        self.smile_ema: Optional[float] = None
        self.mouth_base: Optional[float] = None
        self.lift_base: Optional[float] = None
        self.smile_warm = 0
        self.calib = 0
        self.measure_miss = 0
        self.last_result: Optional[FaceSignals] = None
        self.left_ear_hist: deque = deque(maxlen=3)
        self.right_ear_hist: deque = deque(maxlen=3)
        self.mouth_hist: deque = deque(maxlen=5)
        self.lift_hist: deque = deque(maxlen=5)

        model_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "face_landmarker.task"
        )

        if not model_path.exists():
            raise RuntimeError(
                f"Face Landmarker model not found: {model_path}"
            )

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(
            options
        )

    def reset(self) -> None:
        self.low_ear_frames = 0
        self.smiling = False
        self.cooldown = 0
        self.smile_ema = None
        self.mouth_base = None
        self.lift_base = None
        self.smile_warm = 0
        self.calib = 0
        self.measure_miss = 0
        self.last_result = None
        self.left_ear_hist.clear()
        self.right_ear_hist.clear()
        self.mouth_hist.clear()
        self.lift_hist.clear()

    def close(self) -> None:
        self.landmarker.close()

    def _measure(
        self,
        frame: np.ndarray,
        bbox,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Return left EAR, right EAR, mouth-width ratio, corner lift."""

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        side = int(max(bw, bh) * 1.5)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        rx1 = int(round(cx - side / 2.0))
        ry1 = int(round(cy - side / 2.0))
        rx2 = rx1 + side
        ry2 = ry1 + side

        pad_l = max(0, -rx1)
        pad_t = max(0, -ry1)
        pad_r = max(0, rx2 - width)
        pad_b = max(0, ry2 - height)

        roi = frame[
            max(0, ry1):min(height, ry2),
            max(0, rx1):min(width, rx2),
        ]

        if roi.size == 0:
            return None

        if pad_l or pad_t or pad_r or pad_b:
            roi = cv2.copyMakeBorder(
                roi,
                pad_t,
                pad_b,
                pad_l,
                pad_r,
                cv2.BORDER_REPLICATE,
            )

        roi = cv2.resize(
            roi,
            (192, 192),
            interpolation=cv2.INTER_LINEAR,
        )

        rgb = np.ascontiguousarray(
            cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        )

        result = self.landmarker.detect(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )
        )

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        points = np.array(
            [[p.x * 192.0, p.y * 192.0] for p in lm],
            dtype=np.float32,
        )

        left_ear = eye_aspect_ratio(points, LEFT_EYE)
        right_ear = eye_aspect_ratio(points, RIGHT_EYE)

        left_corner = points[MOUTH_LEFT]
        right_corner = points[MOUTH_RIGHT]
        lip_top = points[LIP_TOP]
        lip_bottom = points[LIP_BOTTOM]

        mouth_width = max(
            distance(left_corner, right_corner),
            1e-6,
        )
        face_width = max(
            distance(points[FACE_LEFT], points[FACE_RIGHT]),
            1e-6,
        )
        mouth_ratio = mouth_width / face_width

        # Positive when the corners sit above the lip opening.
        # Image y grows downward, so a smile makes this larger.
        opening_y = 0.5 * (float(lip_top[1]) + float(lip_bottom[1]))
        corner_y = 0.5 * (float(left_corner[1]) + float(right_corner[1]))
        corner_lift = (opening_y - corner_y) / mouth_width

        return left_ear, right_ear, mouth_ratio, corner_lift

    def _update_smile(
        self,
        mouth_ratio: float,
        corner_lift: float,
    ) -> float:
        """
        Score is how much wider and more lifted the mouth is
        than this person's own resting face. A naturally wide
        mouth stays near 0 until they actually smile.
        """

        if self.mouth_base is None or self.lift_base is None:
            self.mouth_base = mouth_ratio
            self.lift_base = corner_lift
        else:
            if self.calib < 15:
                mouth_follow = 0.35
                lift_follow = 0.35
                self.calib += 1
            else:
                mouth_follow = 0.25 if mouth_ratio < self.mouth_base else 0.004
                lift_follow = 0.25 if corner_lift < self.lift_base else 0.004

            self.mouth_base = (
                (1.0 - mouth_follow) * self.mouth_base
                + mouth_follow * mouth_ratio
            )
            self.lift_base = (
                (1.0 - lift_follow) * self.lift_base
                + lift_follow * corner_lift
            )

        wider = mouth_ratio - float(self.mouth_base)
        raised = corner_lift - float(self.lift_base)

        if self.smile_ema is None:
            self.smile_ema = wider
        else:
            self.smile_ema = 0.65 * self.smile_ema + 0.35 * wider

        score = float(self.smile_ema)

        if self.calib < 12:
            self.smiling = False
            self.smile_warm = 0
            return score

        if self.smiling:
            if score < self.smile_off or raised < -0.01:
                self.smiling = False
                self.smile_warm = 0
            return score

        if score >= self.smile_on and raised >= 0.02:
            self.smile_warm += 1
        else:
            self.smile_warm = 0

        if self.smile_warm >= 3:
            self.smiling = True

        return score

    def analyze(
        self,
        frame: np.ndarray,
        bbox,
    ) -> Optional[FaceSignals]:
        measured = self._measure(frame, bbox)

        if measured is None:
            self.measure_miss += 1

            if (
                self.last_result is not None
                and self.measure_miss <= 3
            ):
                held = self.last_result
                return FaceSignals(
                    ear=held.ear,
                    blink=False,
                    eyes_closed=held.eyes_closed,
                    smile_score=held.smile_score,
                    smiling=held.smiling,
                )

            return None

        self.measure_miss = 0
        left_ear, right_ear, mouth_ratio, corner_lift = measured

        self.left_ear_hist.append(left_ear)
        self.right_ear_hist.append(right_ear)

        left_stable = float(np.median(self.left_ear_hist))
        right_stable = float(np.median(self.right_ear_hist))
        ear = 0.5 * (left_stable + right_stable)

        if self.cooldown > 0:
            self.cooldown -= 1

        blink = False
        eyes_shut = (
            left_stable < self.ear_threshold
            and right_stable < self.ear_threshold
        )

        if eyes_shut:
            self.low_ear_frames += 1
        else:
            if (
                self.cooldown == 0
                and self.blink_min_frames
                <= self.low_ear_frames
                <= self.blink_max_frames
            ):
                blink = True
                self.cooldown = 8

            self.low_ear_frames = 0

        eyes_closed = self.low_ear_frames >= self.closed_frames

        self.mouth_hist.append(mouth_ratio)
        self.lift_hist.append(corner_lift)
        mouth_ratio = float(np.median(self.mouth_hist))
        corner_lift = float(np.median(self.lift_hist))

        smile_score = self._update_smile(mouth_ratio, corner_lift)

        self.last_result = FaceSignals(
            ear=ear,
            blink=blink,
            eyes_closed=eyes_closed,
            smile_score=smile_score,
            smiling=self.smiling,
        )

        return self.last_result