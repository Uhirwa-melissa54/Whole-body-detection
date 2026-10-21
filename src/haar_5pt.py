"""
Haar face detection + practical 5-point landmarks.

Why this works:
- Haar is fast and robust on CPU.
- MediaPipe Face Landmarker confirms a real face and gives stable landmarks.
- We extract ONLY 5 keypoints:
  left_eye, right_eye, nose_tip, mouth_left, mouth_right
- We rebuild bbox from keypoints.
- We reject Haar false positives if MediaPipe doesn't produce landmarks.

Run:

python -m src.haar_5pt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -------------------------
# Data
# -------------------------

@dataclass
class FaceKpsBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32


# -------------------------
# Helpers
# -------------------------

def _similarity_2x3(
    src: np.ndarray,
    dst: np.ndarray,
) -> np.ndarray:
    """
    Deterministic least-squares similarity (Umeyama).

    Same 5 points always produce the same warp, so the aligned
    face does not jitter the way LMEDS does.
    """

    src64 = src.astype(np.float64)
    dst64 = dst.astype(np.float64)

    n, dim = src64.shape

    src_mean = src64.mean(axis=0)
    dst_mean = dst64.mean(axis=0)

    src_demean = src64 - src_mean
    dst_demean = dst64 - dst_mean

    cov = (dst_demean.T @ src_demean) / float(n)

    U, S, Vt = np.linalg.svd(cov)

    d = np.ones((dim,), dtype=np.float64)

    if np.linalg.det(cov) < 0.0:
        d[-1] = -1.0

    rot = U @ np.diag(d) @ Vt

    var = float((src_demean ** 2).sum() / float(n))
    scale = float(np.dot(S, d) / (var + 1e-12))

    trans = dst_mean - scale * (rot @ src_mean)

    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = (scale * rot).astype(np.float32)
    M[:, 2] = trans.astype(np.float32)

    return M


def _estimate_norm_5pt(
    kps_5x2: np.ndarray,
    out_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """
    Build 2x3 affine matrix that maps your 5pts to ArcFace-style template.

    kps order must be:
    [Leye, Reye, Nose, Lmouth, Rmouth]
    """

    k = kps_5x2.astype(np.float32)

    # ArcFace 112x112 template
    dst = np.array(
        [
            [38.2946, 51.6963],  # left eye
            [73.5318, 51.5014],  # right eye
            [56.0252, 71.7366],  # nose
            [41.5493, 92.3655],  # left mouth
            [70.7299, 92.2041],  # right mouth
        ],
        dtype=np.float32,
    )

    out_w, out_h = (
        int(out_size[0]),
        int(out_size[1]),
    )

    if (out_w, out_h) != (112, 112):
        sx = out_w / 112.0
        sy = out_h / 112.0

        dst = dst * np.array(
            [sx, sy],
            dtype=np.float32,
        )

    M = _similarity_2x3(k, dst)

    scale = float(np.linalg.norm(M[:, 0]))

    if not np.isfinite(scale) or scale < 1e-3 or scale > 50.0:
        M = cv2.getAffineTransform(
            np.array(
                [k[0], k[1], k[2]],
                dtype=np.float32,
            ),
            np.array(
                [dst[0], dst[1], dst[2]],
                dtype=np.float32,
            ),
        )

    return M.astype(np.float32)


def align_face_5pt(
    frame_bgr: np.ndarray,
    kps_5x2: np.ndarray,
    out_size: Tuple[int, int] = (112, 112),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        aligned_bgr, M
    """

    M = _estimate_norm_5pt(
        kps_5x2,
        out_size=out_size,
    )

    out_w, out_h = (
        int(out_size[0]),
        int(out_size[1]),
    )

    aligned = cv2.warpAffine(
        frame_bgr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return aligned, M


def _clip_box_xyxy(
    b: np.ndarray,
    W: int,
    H: int,
) -> np.ndarray:

    bb = b.astype(np.float32).copy()

    bb[0] = np.clip(bb[0], 0, W - 1)
    bb[1] = np.clip(bb[1], 0, H - 1)
    bb[2] = np.clip(bb[2], 0, W - 1)
    bb[3] = np.clip(bb[3], 0, H - 1)

    return bb


def _bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15,
) -> np.ndarray:
    """
    Build a face bbox from 5 keypoints with asymmetric padding.
    """

    k = kps.astype(np.float32)

    x_min = float(np.min(k[:, 0]))
    x_max = float(np.max(k[:, 0]))
    y_min = float(np.min(k[:, 1]))
    y_max = float(np.max(k[:, 1]))

    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)

    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w

    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h

    return np.array(
        [x1, y1, x2, y2],
        dtype=np.float32,
    )


def _ema(
    prev: Optional[np.ndarray],
    cur: np.ndarray,
    alpha: float,
) -> np.ndarray:

    if prev is None:
        return cur.astype(np.float32)

    return (
        alpha * prev
        + (1.0 - alpha) * cur
    ).astype(np.float32)


def _kps_span_ok(
    kps: np.ndarray,
    min_eye_dist: float = 12.0,
) -> bool:
    """
    Quick sanity filter on 5pt geometry.
    """

    k = kps.astype(np.float32)

    le, re, no, lm, rm = k

    eye_dist = float(
        np.linalg.norm(re - le)
    )

    if eye_dist < min_eye_dist:
        return False

    # Mouth should generally be below nose.
    if not (
        lm[1] > no[1]
        and rm[1] > no[1]
    ):
        return False

    return True


# -------------------------
# Detector
# -------------------------

class Haar5ptDetector:

    def __init__(
        self,
        haar_xml: Optional[str] = None,
        min_size: Tuple[int, int] = (60, 60),
        smooth_alpha: float = 0.40,
        debug: bool = True,
        detect_max_width: int = 480,
    ):

        self.debug = bool(debug)

        self.min_size = tuple(
            map(int, min_size)
        )

        # Weight on the PREVIOUS keypoints.
        # 0.40 damps jitter without the long lag of 0.80.
        self.smooth_alpha = float(
            smooth_alpha
        )

        # Haar + Face Landmarker run on this width.
        # The warp still uses the full-resolution frame.
        self.detect_max_width = int(
            detect_max_width
        )

        self._hold_frames = 3
        self._miss_reset = 10

        # -------------------------
        # Project root
        # -------------------------

        project_root = Path(__file__).resolve().parent.parent

        # -------------------------
        # Haar cascade
        # -------------------------

        if haar_xml is None:
            haar_xml = str(
                project_root
                / "data"
                / "haarcascade_frontalface_default.xml"
            )

        self.face_cascade = cv2.CascadeClassifier(
            haar_xml
        )

        if self.face_cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade: {haar_xml}"
            )

        # -------------------------
        # MediaPipe Face Landmarker
        # -------------------------

        model_path = (
            project_root
            / "data"
            / "face_landmarker.task"
        )

        if not model_path.exists():
            raise RuntimeError(
                f"Face Landmarker model not found: {model_path}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.mp_face_landmarker = (
            vision.FaceLandmarker.create_from_options(
                options
            )
        )

        # -------------------------
        # 5-point indices
        # -------------------------

        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291

        # -------------------------
        # Tracking state
        # -------------------------

        self._prev_box: Optional[np.ndarray] = None
        self._prev_kps: Optional[np.ndarray] = None

        self._miss = 0
        self._timestamp_ms = 0

    def _resize_for_detect(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:

        height, width = frame_bgr.shape[:2]
        max_w = self.detect_max_width

        if width <= max_w:
            return frame_bgr, 1.0, 1.0

        scale = max_w / float(width)

        small = cv2.resize(
            frame_bgr,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )

        scale_x = small.shape[1] / float(width)
        scale_y = small.shape[0] / float(height)

        return small, scale_x, scale_y

    def _smooth_observation(
        self,
        box: np.ndarray,
        kps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if self._prev_kps is None or self._prev_box is None:
            return (
                box.astype(np.float32),
                kps.astype(np.float32),
            )

        eye = float(
            np.linalg.norm(kps[1] - kps[0])
        ) + 1e-6

        jump = float(
            np.linalg.norm(
                kps.mean(axis=0)
                - self._prev_kps.mean(axis=0)
            )
        )

        # A real teleport (lost track). Snap instead of
        # dragging the old face across the frame.
        if jump > 1.6 * eye:
            return (
                box.astype(np.float32),
                kps.astype(np.float32),
            )

        return (
            _ema(
                self._prev_box,
                box,
                self.smooth_alpha,
            ),
            _ema(
                self._prev_kps,
                kps,
                self.smooth_alpha,
            ),
        )

    def _hold_last(self) -> List[FaceKpsBox]:

        self._miss += 1

        if self._miss > self._miss_reset:
            self._prev_box = None
            self._prev_kps = None
            return []

        if (
            self._prev_kps is None
            or self._prev_box is None
            or self._miss > self._hold_frames
        ):
            return []

        x1, y1, x2, y2 = self._prev_box.tolist()

        return [
            FaceKpsBox(
                x1=int(round(x1)),
                y1=int(round(y1)),
                x2=int(round(x2)),
                y2=int(round(y2)),
                score=1.0,
                kps=self._prev_kps.astype(np.float32),
            )
        ]

    def _haar_faces(
        self,
        gray: np.ndarray,
        min_size: Tuple[int, int],
    ) -> np.ndarray:

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=min_size,
        )

        if faces is None or len(faces) == 0:
            return np.zeros(
                (0, 4),
                dtype=np.int32,
            )

        return faces.astype(np.int32)

    def _facemesh_5pt(
        self,
        frame_bgr: np.ndarray,
    ) -> Optional[np.ndarray]:

        H, W = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        self._timestamp_ms += 33

        result = self.mp_face_landmarker.detect_for_video(
            mp_image,
            self._timestamp_ms,
        )

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]

        idxs = [
            self.IDX_LEFT_EYE,
            self.IDX_RIGHT_EYE,
            self.IDX_NOSE_TIP,
            self.IDX_MOUTH_LEFT,
            self.IDX_MOUTH_RIGHT,
        ]

        pts = []

        for i in idxs:

            p = lm[i]

            pts.append(
                [
                    p.x * W,
                    p.y * H,
                ]
            )

        kps = np.array(
            pts,
            dtype=np.float32,
        )

        # Image-left / image-right. Mouth swap must exchange
        # the two mouth corners only (not a mouth corner and an eye).
        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]

        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 3]]

        return kps

    def detect(
        self,
        frame_bgr: np.ndarray,
        max_faces: int = 1,
    ) -> List[FaceKpsBox]:

        H, W = frame_bgr.shape[:2]

        small, scale_x, scale_y = self._resize_for_detect(
            frame_bgr
        )

        gray = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2GRAY,
        )

        min_px = max(
            24,
            int(round(self.min_size[0] * scale_x)),
        )

        faces = self._haar_faces(
            gray,
            (min_px, min_px),
        )

        haar_box = None

        if faces.shape[0] > 0:

            areas = faces[:, 2] * faces[:, 3]
            i = int(np.argmax(areas))
            haar_box = faces[i].tolist()

        elif self._prev_kps is None:
            return []

        # Landmarks on the small frame. VIDEO mode keeps
        # the face through brief Haar dropouts.
        kps = self._facemesh_5pt(small)

        if kps is None:

            if self.debug and haar_box is not None:
                print(
                    "[haar_5pt] Haar face found but "
                    "Face Landmarker returned none -> reject"
                )

            return self._hold_last()

        if haar_box is not None:

            x, y, w, h = haar_box

            margin = 0.35

            x1m = x - margin * w
            y1m = y - margin * h

            x2m = x + (1.0 + margin) * w
            y2m = y + (1.0 + margin) * h

            inside = (
                (kps[:, 0] >= x1m)
                & (kps[:, 0] <= x2m)
                & (kps[:, 1] >= y1m)
                & (kps[:, 1] <= y2m)
            )

            if inside.mean() < 0.60:

                if self.debug:
                    print(
                        "[haar_5pt] Face Landmarker points "
                        "not consistent with Haar box -> reject"
                    )

                return self._hold_last()

            min_eye = max(8.0, 0.18 * float(w))

        else:
            min_eye = 8.0

        if not _kps_span_ok(
            kps,
            min_eye_dist=min_eye,
        ):

            if self.debug:
                print(
                    "[haar_5pt] 5pt geometry sanity "
                    "failed -> reject"
                )

            return self._hold_last()

        kps = kps.astype(np.float32)
        kps[:, 0] /= float(scale_x)
        kps[:, 1] /= float(scale_y)

        # Build centered bbox in full-frame pixels.
        box = _bbox_from_5pt(
            kps,
            pad_x=0.55,
            pad_y_top=0.85,
            pad_y_bot=1.15,
        )

        box = _clip_box_xyxy(
            box,
            W,
            H,
        )

        box_s, kps_s = self._smooth_observation(
            box,
            kps,
        )

        self._miss = 0
        self._prev_box = box_s.copy()
        self._prev_kps = kps_s.copy()

        x1, y1, x2, y2 = box_s.tolist()

        # Haar doesn't provide probability.
        score = 1.0

        return [
            FaceKpsBox(
                x1=int(round(x1)),
                y1=int(round(y1)),
                x2=int(round(x2)),
                y2=int(round(y2)),
                score=float(score),
                kps=kps_s.astype(np.float32),
            )
        ][:max_faces]


# -------------------------
# Demo
# -------------------------

def main():

    cap = cv2.VideoCapture(0)

    det = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.40,
        debug=True,
    )

    print(
        "Haar + 5pt (Face Landmarker) test. "
        "Press q to quit."
    )

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        faces = det.detect(
            frame,
            max_faces=1,
        )

        vis = frame.copy()

        if faces:

            f = faces[0]

            cv2.rectangle(
                vis,
                (f.x1, f.y1),
                (f.x2, f.y2),
                (0, 255, 0),
                2,
            )

            for (x, y) in f.kps.astype(int):

                cv2.circle(
                    vis,
                    (int(x), int(y)),
                    3,
                    (0, 255, 0),
                    -1,
                )

            cv2.putText(
                vis,
                "OK",
                (f.x1, max(0, f.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        else:

            cv2.putText(
                vis,
                "no face",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        cv2.imshow(
            "haar_5pt",
            vis,
        )

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()