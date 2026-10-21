"""CPU-only whole-person tracking with honest face-association state.

The target is initially the largest person whose detected face maps to its
upper body.  Face loss does not become a claim that the face is still seen:
the tracker moves through CONFIRMED -> TEMPORARY -> UNKNOWN while confidence
decays exponentially.  It is intentionally a camera-only evaluator; no motor,
MQTT, enrollment database, or hardware is required.

Run:
    python -m src.whole_person_tracking --camera 0 --mirror

Keys: Q quit, F fullscreen/windowed, R clear current target and reacquire.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.align import align_face_5pt
from src.recognize import ArcFaceEmbedderONNX, FaceDBMatcher, HaarFaceMesh5pt, load_db_npz


class AssociationState(str, Enum):
    CONFIRMED = "CONFIRMED"
    TEMPORARY = "TEMPORARY"
    UNKNOWN = "UNKNOWN"


@dataclass
class BodyTrack:
    box: tuple[int, int, int, int]
    velocity: np.ndarray
    confidence: float
    state: AssociationState
    last_face_time: float
    last_body_time: float


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return intersection / float(area_a + area_b - intersection)


def box_center(box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array(((x1 + x2) / 2.0, (y1 + y2) / 2.0), dtype=np.float32)


def clamp_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.round().astype(int)
    return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)


def face_in_body(face: tuple[int, int, int, int], body: tuple[int, int, int, int]) -> bool:
    """Map face centre to the upper 65% of the person box."""
    fx, fy = box_center(face)
    x1, y1, x2, y2 = body
    return x1 <= fx <= x2 and y1 <= fy <= y1 + 0.65 * max(1, y2 - y1)


class WholePersonTracker:
    def __init__(self, temporary_seconds: float = 2.5, confidence_half_life: float = 1.0):
        self.temporary_seconds = temporary_seconds
        self.confidence_half_life = confidence_half_life
        self.track: Optional[BodyTrack] = None

    def reset(self) -> None:
        self.track = None

    def _decay(self, elapsed: float) -> float:
        return math.exp(-math.log(2.0) * max(0.0, elapsed) / self.confidence_half_life)

    def _match_body(self, bodies: list[tuple[int, int, int, int]], dt: float) -> Optional[tuple[int, int, int, int]]:
        if not self.track or not bodies:
            return None
        predicted = np.array(self.track.box, dtype=np.float32)
        shift = np.tile(self.track.velocity * dt, 2)
        predicted += shift
        predicted_box = tuple(predicted.tolist())
        candidate = max(bodies, key=lambda body: box_iou(predicted_box, body))
        return candidate if box_iou(predicted_box, candidate) >= 0.08 else None

    def update(
        self,
        bodies: list[tuple[int, int, int, int]],
        faces: list[tuple[int, int, int, int]],
        now: float,
        dt: float,
    ) -> Optional[BodyTrack]:
        if self.track is None:
            pairs = [(body, face) for body in bodies for face in faces if face_in_body(face, body)]
            if not pairs:
                return None
            # Acquire the largest valid whole-person/face pair, not merely a face.
            body, _face = max(pairs, key=lambda pair: (pair[0][2] - pair[0][0]) * (pair[0][3] - pair[0][1]))
            self.track = BodyTrack(body, np.zeros(2, dtype=np.float32), 1.0,
                                   AssociationState.CONFIRMED, now, now)
            return self.track

        body = self._match_body(bodies, dt)
        if body is not None:
            old_center = box_center(self.track.box)
            new_center = box_center(body)
            self.track.velocity = 0.55 * self.track.velocity + 0.45 * ((new_center - old_center) / max(dt, 1e-3))
            self.track.box = body
            self.track.last_body_time = now

        face_confirmed = body is not None and any(face_in_body(face, body) for face in faces)
        if face_confirmed:
            self.track.state = AssociationState.CONFIRMED
            self.track.confidence = 1.0
            self.track.last_face_time = now
        else:
            face_gap = now - self.track.last_face_time
            if body is not None and face_gap <= self.temporary_seconds:
                self.track.state = AssociationState.TEMPORARY
                self.track.confidence = self._decay(face_gap)
            else:
                self.track.state = AssociationState.UNKNOWN
                self.track.confidence = 0.0
        return self.track


def detect_people_hog(hog: cv2.HOGDescriptor, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    rectangles, weights = hog.detectMultiScale(
        frame, winStride=(8, 8), padding=(8, 8), scale=1.05
    )
    height, width = frame.shape[:2]
    return [
        (max(0, x), max(0, y), min(width - 1, x + w), min(height - 1, y + h))
        for (x, y, w, h), weight in zip(rectangles, weights)
        if float(weight) >= 0.2
    ]


class YOLOPersonDetector:
    """CPU YOLO detector restricted to COCO's `person` class (0)."""
    def __init__(self, model_name: str, confidence: float, image_size: int):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO is not installed. Run: .\\.venv\\Scripts\\python.exe -m pip install -r requirements-yolo.txt"
            ) from exc
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        result = self.model.predict(
            frame, classes=[0], conf=self.confidence, imgsz=self.image_size,
            device="cpu", verbose=False,
        )[0]
        if result.boxes is None:
            return []
        height, width = frame.shape[:2]
        return [
            clamp_box(np.asarray(box, dtype=np.float32), width, height)
            for box in result.boxes.xyxy.cpu().numpy()
        ]


def draw_text(frame: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.62) -> None:
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only whole-person face-association evaluator")
    parser.add_argument("--camera", type=int, default=0)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--target", help="enrolled identity to acquire and track")
    selection.add_argument("--any-face", action="store_true",
                           help="acquire any face-and-body pair; evaluation mode only")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--detector", choices=("yolo", "hog"), default="yolo",
                        help="whole-person detector; YOLO is default, HOG is fallback")
    parser.add_argument("--yolo-model", default="yolo11n.pt",
                        help="Ultralytics model; first use downloads weights")
    parser.add_argument("--person-confidence", type=float, default=0.35)
    parser.add_argument("--yolo-image-size", type=int, default=416)
    parser.add_argument("--body-every", type=int, default=3,
                        help="run whole-person detection every N frames; faces run every frame")
    parser.add_argument("--identity-similarity", type=float, default=0.58,
                        help="minimum cosine similarity for --target confirmation")
    parser.add_argument("--temporary-seconds", type=float, default=2.5)
    parser.add_argument("--half-life", type=float, default=1.0,
                        help="confidence half-life during temporary face loss")
    args = parser.parse_args()
    if args.temporary_seconds <= 0 or args.half_life <= 0:
        parser.error("temporary-seconds and half-life must be positive")
    if not 0 < args.person_confidence <= 1 or args.yolo_image_size < 160 or args.body_every < 1:
        parser.error("person-confidence must be in (0, 1], yolo-image-size at least 160, and body-every at least 1")
    if not 0 < args.identity_similarity <= 1:
        parser.error("identity-similarity must be in (0, 1]")

    embedder = matcher = None
    if args.target:
        matcher = FaceDBMatcher(load_db_npz(Path("data/db/face_db.npz")))
        if args.target not in matcher.db:
            parser.error(f"Unknown enrolled target {args.target!r}. Available: {', '.join(sorted(matcher.db))}")
        embedder = ArcFaceEmbedderONNX("models/embedder_arcface.onnx", input_size=(112, 112), debug=False)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {args.camera} is not available")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    hog = None
    yolo = None
    if args.detector == "yolo":
        yolo = YOLOPersonDetector(args.yolo_model, args.person_confidence, args.yolo_image_size)
        print(f"YOLO person detection: {args.yolo_model} on CPU")
    else:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print("OpenCV HOG person detection on CPU")
    faces = HaarFaceMesh5pt(min_size=(40, 40), debug=False)
    tracker = WholePersonTracker(args.temporary_seconds, args.half_life)
    window = "Whole Person Face Association"
    fullscreen = not args.windowed
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    last_time = time.monotonic()
    frame_index = 0
    body_boxes: list[tuple[int, int, int, int]] = []

    try:
        while True:
            # Prefer the newest frame. Slow CPU detection otherwise leaves the
            # display several frames behind the person and looks like blur.
            for _ in range(2):
                cap.grab()
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            now = time.monotonic()
            dt = min(max(now - last_time, 1 / 120), 0.2)
            last_time = now
            frame_index += 1
            if frame_index % args.body_every == 1 or not body_boxes:
                body_boxes = yolo.detect(frame) if yolo else detect_people_hog(hog, frame)
            detected_faces = faces.detect(frame, max_faces=5, coarse=True)
            face_boxes = []
            for face in detected_faces:
                box = (face.x1, face.y1, face.x2, face.y2)
                if args.target:
                    aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
                    if aligned is None or not aligned.size:
                        continue
                    similarity = float(np.dot(embedder.embed(aligned).reshape(-1), matcher.db[args.target].reshape(-1)))
                    if similarity < args.identity_similarity:
                        continue
                face_boxes.append(box)
            track = tracker.update(body_boxes, face_boxes, now, dt)
            view = frame.copy()
            for body in body_boxes:
                cv2.rectangle(view, body[:2], body[2:], (100, 100, 100), 1)
            for face in face_boxes:
                cv2.rectangle(view, face[:2], face[2:], (180, 180, 180), 1)

            if track:
                color = {AssociationState.CONFIRMED: (0, 220, 0), AssociationState.TEMPORARY: (0, 190, 255), AssociationState.UNKNOWN: (0, 0, 255)}[track.state]
                cv2.rectangle(view, track.box[:2], track.box[2:], color, 3)
                cx, cy = box_center(track.box)
                error = (cx - view.shape[1] / 2) / (view.shape[1] / 2)
                angle = int(round(90 + np.clip(error, -1, 1) * 70))
                identity = args.target if args.target else "ANY FACE"
                draw_text(view, f"{identity}: {track.state.value} confidence={track.confidence:.2f}", (12, 30), color, 0.68)
                draw_text(view, f"body-centering angle={angle} deg", (12, 58), (255, 255, 0), 0.56)
                cv2.line(view, (view.shape[1] // 2, 0), (view.shape[1] // 2, view.shape[0]), (0, 180, 255), 1)
            else:
                identity = args.target if args.target else "ANY FACE"
                draw_text(view, f"{identity}: UNKNOWN, waiting for matching face + whole person", (12, 30), (0, 120, 255), 0.62)

            draw_text(view, f"{args.detector.upper()} | Q quit | F fullscreen | R reacquire", (12, view.shape[0] - 18), (255, 255, 255), 0.46)
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                tracker.reset()
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    finally:
        cap.release()
        faces.landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
