
"""
Multi-face recognition (CPU-friendly) using the stable pipeline:

Haar (multi-face)
    -> MediaPipe Face Landmarker 5pt (per-face ROI)
    -> align_face_5pt (112x112)
    -> ArcFace ONNX embedding
    -> cosine distance to DB
    -> label each face.

Run:
    python -m src.recognize

Keys:
    q : quit
    r : reload DB from disk (data/db/face_db.npz)
    +/- : adjust threshold (distance) live
    d : toggle debug overlay

Notes:
- We run MediaPipe Face Landmarker on EACH Haar face ROI.
- DB is expected from enroll:
    data/db/face_db.npz
- Distance definition:
    cosine_distance = 1 - cosine_similarity
- Since embeddings are L2-normalized:
    cosine_similarity = dot(a,b)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .haar_5pt import align_face_5pt


# -------------------------
# Data
# -------------------------

@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32 in FULL-frame coords


@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


# -------------------------
# Math helpers
# -------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)

    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)


def _clip_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    W: int,
    H: int,
) -> Tuple[int, int, int, int]:

    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))

    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def _bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15,
) -> np.ndarray:
    """
    Build a nicer face-like bbox from 5 points with asymmetric padding.

    kps: (5,2) in full-frame coords
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


def _kps_span_ok(
    kps: np.ndarray,
    min_eye_dist: float,
) -> bool:
    """
    Minimal geometry sanity:

    - eyes not collapsed
    - mouth generally below nose
    """

    k = kps.astype(np.float32)

    le, re, no, lm, rm = k

    eye_dist = float(np.linalg.norm(re - le))

    if eye_dist < float(min_eye_dist):
        return False

    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False

    return True


# -------------------------
# DB helpers
# -------------------------

def load_db_npz(
    db_path: Path,
) -> Dict[str, np.ndarray]:

    if not db_path.exists():
        return {}

    data = np.load(
        str(db_path),
        allow_pickle=True,
    )

    out: Dict[str, np.ndarray] = {}

    for k in data.files:
        out[k] = np.asarray(
            data[k],
            dtype=np.float32,
        ).reshape(-1)

    return out


# -------------------------
# Embedder
# -------------------------

class ArcFaceEmbedderONNX:
    """
    ArcFace-style ONNX embedder.

    Input:
        112x112 BGR
        -> RGB
        -> (x - 127.5) / 128
        -> NCHW float32

    Output:
        embedding vector
    """

    def __init__(
        self,
        model_path: str = "models/embedder_arcface.onnx",
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
    ):

        self.model_path = model_path

        self.in_w = int(input_size[0])
        self.in_h = int(input_size[1])

        self.debug = bool(debug)

        self.sess = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

        if self.debug:
            print("[embed] model:", model_path)

            print(
                "[embed] input:",
                self.sess.get_inputs()[0].name,
                self.sess.get_inputs()[0].shape,
                self.sess.get_inputs()[0].type,
            )

            print(
                "[embed] output:",
                self.sess.get_outputs()[0].name,
                self.sess.get_outputs()[0].shape,
                self.sess.get_outputs()[0].type,
            )

    def _preprocess(
        self,
        aligned_bgr_112: np.ndarray,
    ) -> np.ndarray:

        img = aligned_bgr_112

        if (
            img.shape[1] != self.in_w
            or img.shape[0] != self.in_h
        ):
            img = cv2.resize(
                img,
                (self.in_w, self.in_h),
                interpolation=cv2.INTER_LINEAR,
            )

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB,
        ).astype(np.float32)

        rgb = (rgb - 127.5) / 128.0

        x = np.transpose(
            rgb,
            (2, 0, 1),
        )[None, ...]

        return x.astype(np.float32)

    @staticmethod
    def _l2_normalize(
        v: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:

        v = v.astype(np.float32).reshape(-1)

        n = float(
            np.linalg.norm(v) + eps
        )

        return (v / n).astype(np.float32)

    def embed(
        self,
        aligned_bgr_112: np.ndarray,
    ) -> np.ndarray:

        x = self._preprocess(
            aligned_bgr_112
        )

        y = self.sess.run(
            [self.out_name],
            {self.in_name: x},
        )[0]

        emb = np.asarray(
            y,
            dtype=np.float32,
        ).reshape(-1)

        return self._l2_normalize(emb)


# -------------------------
# Multi-face Haar + MediaPipe 5pt
# -------------------------

class HaarFaceMesh5pt:

    def __init__(
        self,
        haar_xml: Optional[str] = None,
        min_size: Tuple[int, int] = (70, 70),
        debug: bool = False,
    ):

        self.debug = bool(debug)

        self.min_size = tuple(
            map(int, min_size)
        )

        # -------------------------
        # Haar cascade
        # -------------------------

        if haar_xml is None:
            project_root = (
                Path(__file__).resolve().parent.parent
            )

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

        project_root = (
            Path(__file__).resolve().parent.parent
        )

        model_path = (
            project_root
            / "data"
            / "face_landmarker.task"
        )

        if not model_path.exists():
            raise RuntimeError(
                f"MediaPipe model not found: {model_path}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )

        # IMAGE mode: each crop is its own picture.
        # VIDEO mode treats the next crop as the next
        # frame of the same shot, which makes landmarks jump.
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )

        self.landmarker = (
            vision.FaceLandmarker.create_from_options(
                options
            )
        )

        # 5pt indices
        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291

    def _haar_faces(
        self,
        gray: np.ndarray,
        min_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:

        if min_size is None:
            min_size = self.min_size

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=tuple(map(int, min_size)),
        )

        if faces is None or len(faces) == 0:
            return np.zeros(
                (0, 4),
                dtype=np.int32,
            )

        return faces.astype(np.int32)

    def _roi_facemesh_5pt(
        self,
        roi_bgr: np.ndarray,
    ) -> Optional[np.ndarray]:

        H, W = roi_bgr.shape[:2]

        if H < 20 or W < 20:
            return None

        rgb = cv2.cvtColor(
            roi_bgr,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )

        result = self.landmarker.detect(mp_image)

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

        # enforce left/right ordering
        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]

        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 3]]

        return kps

    def detect(
        self,
        frame_bgr: np.ndarray,
        max_faces: int = 5,
        coarse: bool = False,
    ) -> List[FaceDet]:

        H, W = frame_bgr.shape[:2]

        max_w = 640

        if W > max_w:
            small = cv2.resize(
                frame_bgr,
                (
                    max_w,
                    max(1, int(round(H * max_w / float(W)))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame_bgr

        scale_x = small.shape[1] / float(W)
        scale_y = small.shape[0] / float(H)
        sH, sW = small.shape[:2]

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

        if faces.shape[0] == 0:
            return []

        # sort by area desc, keep top max_faces
        areas = faces[:, 2] * faces[:, 3]

        order = np.argsort(areas)[::-1]

        faces = faces[order][:max_faces]

        out: List[FaceDet] = []

        for (x, y, w, h) in faces:

            # expand ROI a bit for Face Landmarker stability
            mx, my = (
                0.25 * w,
                0.35 * h,
            )

            rx1, ry1, rx2, ry2 = _clip_xyxy(
                x - mx,
                y - my,
                x + w + mx,
                y + h + my,
                sW,
                sH,
            )

            roi = small[
                ry1:ry2,
                rx1:rx2,
            ]

            kps_roi = self._roi_facemesh_5pt(
                roi
            )

            fx = float(x) / scale_x
            fy = float(y) / scale_y
            fw = float(w) / scale_x
            fh = float(h) / scale_y

            landmarks_ok = kps_roi is not None

            if landmarks_ok:
                kps = kps_roi.copy()
                kps[:, 0] = (kps[:, 0] + float(rx1)) / scale_x
                kps[:, 1] = (kps[:, 1] + float(ry1)) / scale_y

                landmarks_ok = _kps_span_ok(
                    kps,
                    min_eye_dist=max(
                        10.0,
                        0.18 * fw,
                    ),
                )

            if not landmarks_ok:
                if not coarse:
                    continue

                # Keep the Haar box so a brief landmark miss
                # does not look like the person left.
                kps = np.array(
                    [
                        [fx + 0.30 * fw, fy + 0.40 * fh],
                        [fx + 0.70 * fw, fy + 0.40 * fh],
                        [fx + 0.50 * fw, fy + 0.62 * fh],
                        [fx + 0.36 * fw, fy + 0.78 * fh],
                        [fx + 0.64 * fw, fy + 0.78 * fh],
                    ],
                    dtype=np.float32,
                )

                x1, y1, x2, y2 = _clip_xyxy(
                    fx - 0.08 * fw,
                    fy - 0.15 * fh,
                    fx + 1.08 * fw,
                    fy + 1.12 * fh,
                    W,
                    H,
                )

                out.append(
                    FaceDet(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        score=0.4,
                        kps=kps,
                    )
                )
                continue

            # build bbox from keypoints
            bb = _bbox_from_5pt(
                kps,
                pad_x=0.55,
                pad_y_top=0.85,
                pad_y_bot=1.15,
            )

            x1, y1, x2, y2 = _clip_xyxy(
                bb[0],
                bb[1],
                bb[2],
                bb[3],
                W,
                H,
            )

            out.append(
                FaceDet(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=1.0,
                    kps=kps.astype(
                        np.float32
                    ),
                )
            )

        return out


# -------------------------
# Matcher
# -------------------------

class FaceDBMatcher:

    def __init__(
        self,
        db: Dict[str, np.ndarray],
        dist_thresh: float = 0.34,
    ):

        self.db = db
        self.dist_thresh = float(
            dist_thresh
        )

        # pre-stack for speed
        self._names: List[str] = []
        self._mat: Optional[np.ndarray] = None

        self._rebuild()

    def _rebuild(self):

        self._names = sorted(
            self.db.keys()
        )

        if self._names:

            self._mat = np.stack(
                [
                    self.db[n]
                    .reshape(-1)
                    .astype(np.float32)
                    for n in self._names
                ],
                axis=0,
            )

        else:
            self._mat = None

    def reload_from(
        self,
        path: Path,
    ):

        self.db = load_db_npz(path)

        self._rebuild()

    def match(
        self,
        emb: np.ndarray,
    ) -> MatchResult:

        if (
            self._mat is None
            or len(self._names) == 0
        ):

            return MatchResult(
                name=None,
                distance=1.0,
                similarity=0.0,
                accepted=False,
            )

        e = emb.reshape(
            1,
            -1,
        ).astype(np.float32)

        # cosine similarity since both sides
        # are normalized: sim = dot
        sims = (
            self._mat @ e.T
        ).reshape(-1)

        best_i = int(
            np.argmax(sims)
        )

        best_sim = float(
            sims[best_i]
        )

        best_dist = 1.0 - best_sim

        ok = (
            best_dist
            <= self.dist_thresh
        )

        return MatchResult(
            name=(
                self._names[best_i]
                if ok
                else None
            ),
            distance=float(
                best_dist
            ),
            similarity=float(
                best_sim
            ),
            accepted=bool(ok),
        )


# -------------------------
# Demo
# -------------------------

def main():

    db_path = Path(
        "data/db/face_db.npz"
    )

    det = HaarFaceMesh5pt(
        min_size=(70, 70),
        debug=False,
    )

    embedder = ArcFaceEmbedderONNX(
        model_path=(
            "models/embedder_arcface.onnx"
        ),
        input_size=(112, 112),
        debug=False,
    )

    db = load_db_npz(
        db_path
    )

    matcher = FaceDBMatcher(
        db=db,
        dist_thresh=0.34,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Camera not available"
        )

    print(
        "Recognize (multi-face). "
        "q=quit, r=reload DB, "
        "+/- threshold, "
        "d=debug overlay"
    )

    t0 = time.time()
    frames = 0
    fps: Optional[float] = None
    show_debug = False

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        faces = det.detect(
            frame,
            max_faces=5,
        )

        vis = frame.copy()

        # compute FPS
        frames += 1

        dt = time.time() - t0

        if dt >= 1.0:

            fps = frames / dt

            frames = 0

            t0 = time.time()

        # draw + recognize each face
        h, w = vis.shape[:2]

        thumb = 112
        pad = 8

        x0 = w - thumb - pad
        y0 = 80

        shown = 0

        for i, f in enumerate(faces):

            # draw bbox
            cv2.rectangle(
                vis,
                (f.x1, f.y1),
                (f.x2, f.y2),
                (0, 255, 0),
                2,
            )

            # draw 5 landmarks
            for (x, y) in f.kps.astype(
                int
            ):

                cv2.circle(
                    vis,
                    (int(x), int(y)),
                    2,
                    (0, 255, 0),
                    -1,
                )

            # align -> embed -> match
            aligned, _ = align_face_5pt(
                frame,
                f.kps,
                out_size=(112, 112),
            )

            emb = embedder.embed(
                aligned
            )

            mr = matcher.match(
                emb
            )

            # label
            label = (
                mr.name
                if mr.name is not None
                else "Unknown"
            )

            line1 = f"{label}"

            line2 = (
                f"dist={mr.distance:.3f} "
                f"sim={mr.similarity:.3f}"
            )

            # known = green, unknown = red
            color = (
                (0, 255, 0)
                if mr.accepted
                else (0, 0, 255)
            )

            cv2.putText(
                vis,
                line1,
                (
                    f.x1,
                    max(
                        0,
                        f.y1 - 28,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            cv2.putText(
                vis,
                line2,
                (
                    f.x1,
                    max(
                        0,
                        f.y1 - 6,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            # aligned preview thumbnails
            if (
                y0 + thumb <= h
                and shown < 4
            ):

                vis[
                    y0:y0 + thumb,
                    x0:x0 + thumb,
                ] = aligned

                cv2.putText(
                    vis,
                    f"{i + 1}:{label}",
                    (
                        x0,
                        y0 - 6,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

                y0 += thumb + pad

                shown += 1

            if show_debug:

                dbg = (
                    f"kpsLeye="
                    f"({f.kps[0, 0]:.0f},"
                    f"{f.kps[0, 1]:.0f})"
                )

                cv2.putText(
                    vis,
                    dbg,
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        # overlay header
        header = (
            f"IDs={len(matcher._names)} "
            f"thr(dist)="
            f"{matcher.dist_thresh:.2f}"
        )

        if fps is not None:
            header += (
                f" fps={fps:.1f}"
            )

        cv2.putText(
            vis,
            header,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )

        cv2.imshow(
            "recognize_new",
            vis,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):

            matcher.reload_from(
                db_path
            )

            print(
                f"[recognize] "
                f"reloaded DB: "
                f"{len(matcher._names)} "
                f"identities"
            )

        elif key in (
            ord("+"),
            ord("="),
        ):

            matcher.dist_thresh = float(
                min(
                    1.20,
                    matcher.dist_thresh
                    + 0.01,
                )
            )

            print(
                f"[recognize] "
                f"thr(dist)="
                f"{matcher.dist_thresh:.2f} "
                f"(sim~"
                f"{1.0 - matcher.dist_thresh:.2f})"
            )

        elif key == ord("-"):

            matcher.dist_thresh = float(
                max(
                    0.05,
                    matcher.dist_thresh
                    - 0.01,
                )
            )

            print(
                f"[recognize] "
                f"thr(dist)="
                f"{matcher.dist_thresh:.2f} "
                f"(sim~"
                f"{1.0 - matcher.dist_thresh:.2f})"
            )

        elif key == ord("d"):

            show_debug = not show_debug

            print(
                f"[recognize] "
                f"debug overlay: "
                f"{'ON' if show_debug else 'OFF'}"
            )

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
