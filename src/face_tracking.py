# src/face_tracking.py

import argparse
import time

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.align import align_face_5pt
from src.face_signals import FaceSignalExtractor
from src.recognize import (
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    HaarFaceMesh5pt,
    load_db_npz,
)


class LockState(Enum):
    SEARCHING = auto()
    LOCKED = auto()
    LOST = auto()


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return inter / float(area_a + area_b - inter)


def center(box):
    x1, y1, x2, y2 = box

    return np.array(
        [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
        dtype=np.float32,
    )


@dataclass
class TrackingSignal:
    error_x: float
    error_y: float
    horizontal: str
    vertical: str


class LockedFaceTracker:
    def __init__(
        self,
        target_name: Optional[str],
        detector,
        embedder,
        matcher,
        verify_every: int = 15,
        lost_timeout: int = 24,
        ema_alpha: float = 0.35,
        dead_zone: float = 0.08,
    ):
        self.target_name = target_name
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.verify_every = verify_every
        self.lost_timeout = lost_timeout
        self.ema_alpha = ema_alpha
        self.dead_zone = dead_zone
        self.enter_zone = 0.16
        # Stay LOCKED through detection gaps this long.
        self.coast_s = 3.0
        # Similarity to the enrolled face.
        # Acquire is stricter. A weak frame does not unlock.
        self.acquire_sim = 0.58
        self.keep_sim = 0.50
        self.reject_sim = 0.38
        self.verify_fail_limit = 8

        self.state = LockState.SEARCHING
        self.last_box = None
        self.last_face = None
        self.smooth_box = None
        self.smooth_center = None
        self.horizontal = "CENTER"
        self.vertical = "CENTER"

        self.miss_since = None
        self.verify_fails = 0
        self.frame_index = 0

    @staticmethod
    def box(face):
        return (face.x1, face.y1, face.x2, face.y2)

    def similarity_to_target(self, frame, face) -> float:
        template = self.matcher.db.get(self.target_name)

        if template is None:
            return 0.0

        aligned, _ = align_face_5pt(
            frame,
            face.kps,
            out_size=(112, 112),
        )

        emb = self.embedder.embed(aligned).reshape(-1)
        ref = np.asarray(template, dtype=np.float32).reshape(-1)

        return float(np.dot(emb, ref))

    def acquire(self, frame, faces):
        if self.target_name is None:
            candidates = [face for face in faces if face.score >= 1.0]
            return max(candidates, key=lambda face: (face.x2 - face.x1) * (face.y2 - face.y1), default=None)
        best = None
        best_similarity = self.acquire_sim

        for face in faces:
            # Approximate boxes are for holding a lock, not starting one.
            if face.score < 1.0:
                continue

            similarity = self.similarity_to_target(frame, face)

            if similarity > best_similarity:
                best = face
                best_similarity = similarity

        return best

    def associate(self, faces):
        if self.smooth_box is None or not faces:
            return None

        ref = tuple(float(v) for v in self.smooth_box.tolist())
        last_center = center(ref)

        last_diag = max(
            np.linalg.norm(
                np.array(
                    [ref[2] - ref[0], ref[3] - ref[1]],
                    dtype=np.float32,
                )
            ),
            1.0,
        )

        ranked = []

        for face in faces:
            box = self.box(face)
            overlap = iou(ref, box)
            displacement = (
                np.linalg.norm(center(box) - last_center)
                / last_diag
            )

            # Real landmarks beat a Haar-only box at the same place.
            score = overlap - 0.25 * displacement + 0.15 * face.score
            ranked.append((score, overlap, displacement, face))

        _score, overlap, displacement, candidate = max(
            ranked,
            key=lambda item: item[0],
        )

        if overlap > 0.05 or displacement < 0.75:
            return candidate

        return None

    def _remember(self, face):
        raw = np.array(self.box(face), dtype=np.float32)

        if self.smooth_box is None:
            self.smooth_box = raw
        else:
            width = max(1.0, float(raw[2] - raw[0]))
            jump = float(
                np.linalg.norm(
                    center(raw) - center(self.smooth_box)
                )
            )

            if jump > 0.9 * width:
                self.smooth_box = raw
            else:
                a = self.ema_alpha
                self.smooth_box = (
                    a * raw + (1.0 - a) * self.smooth_box
                )

        self.last_face = face
        self.last_box = self.box(face)
        self.smooth_center = center(self.smooth_box)

    def display_box(self):
        if self.smooth_box is None:
            return self.last_box

        x1, y1, x2, y2 = self.smooth_box.tolist()

        return (
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2)),
        )

    def _drop_lock(self):
        self.state = LockState.SEARCHING
        self.last_box = None
        self.last_face = None
        self.smooth_box = None
        self.smooth_center = None
        self.horizontal = "CENTER"
        self.vertical = "CENTER"
        self.verify_fails = 0
        self.miss_since = None

    def update(self, frame):
        self.frame_index += 1

        faces = self.detector.detect(
            frame,
            max_faces=3,
            coarse=True,
        )

        if self.state == LockState.SEARCHING:
            candidate = self.acquire(frame, faces)
        else:
            candidate = self.associate(faces)

        # A bad embedding no longer clears the lock.
        # Only a long run of "this is someone else" does.
        if (
            self.target_name is not None
            and candidate is not None
            and candidate.score >= 1.0
            and self.state == LockState.LOCKED
            and self.frame_index % self.verify_every == 0
        ):
            similarity = self.similarity_to_target(
                frame,
                candidate,
            )

            if similarity >= self.keep_sim:
                self.verify_fails = 0
            elif similarity < self.reject_sim:
                self.verify_fails += 1

            if self.verify_fails >= self.verify_fail_limit:
                self._drop_lock()
                return None, None

        if candidate is None:
            now = time.monotonic()

            if self.last_face is not None:
                if self.miss_since is None:
                    self.miss_since = now

                if (now - self.miss_since) < self.coast_s:
                    self.state = LockState.LOCKED
                    return (
                        self.last_face,
                        self.position_signal(frame.shape),
                    )

            self._drop_lock()
            return None, None

        self.miss_since = None
        self.state = LockState.LOCKED
        self._remember(candidate)

        return candidate, self.position_signal(frame.shape)

    def _axis(self, value, state, negative, positive):
        if state == negative:
            if value < -self.dead_zone:
                return negative
            return "CENTER"

        if state == positive:
            if value > self.dead_zone:
                return positive
            return "CENTER"

        if value < -self.enter_zone:
            return negative

        if value > self.enter_zone:
            return positive

        return "CENTER"

    def position_signal(self, shape) -> TrackingSignal:
        height, width = shape[:2]

        ex = float(
            (self.smooth_center[0] - width / 2.0)
            / (width / 2.0)
        )

        ey = float(
            (self.smooth_center[1] - height / 2.0)
            / (height / 2.0)
        )

        self.horizontal = self._axis(
            ex,
            self.horizontal,
            "LEFT",
            "RIGHT",
        )
        self.vertical = self._axis(
            ey,
            self.vertical,
            "UP",
            "DOWN",
        )

        return TrackingSignal(
            ex,
            ey,
            self.horizontal,
            self.vertical,
        )


def draw_label(
    frame,
    text,
    xy,
    color,
    scale=0.62,
):
    cv2.putText(
        frame,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser()

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--target",
        help="enrolled identity to lock",
    )

    selection.add_argument("--any-face", action="store_true",
                           help="track a face without enrollment")
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="camera index (default: 1 for external USB camera)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.34,
    )

    parser.add_argument("--camera-only", action="store_true", help="test camera without motor communication")
    parser.add_argument("--mirror", action="store_true",
                        help="mirror the camera image horizontally before tracking")
    parser.add_argument("--windowed", action="store_true",
                        help="start in a normal resizable window instead of full screen")
    parser.add_argument("--broker", default="127.0.0.1",
                        help="MQTT broker address; default is this PC")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mode", choices=("pointer", "pan"), default="pointer",
                        help="pointer: fixed camera, needle displays face position; pan: camera mounted on motor")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--min-angle", type=int, default=20)
    parser.add_argument("--max-angle", type=int, default=160)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=45.0, help="pan speed in degrees/second at full error")
    parser.add_argument("--smoothing", type=float, default=0.15, help="smoothing time constant in seconds; 0 disables")
    args = parser.parse_args()
    if not 20 <= args.min_angle < 90 < args.max_angle <= 160:
        parser.error("angles must satisfy 20 <= min < 90 < max <= 160")
    if not np.isfinite([args.gain, args.speed, args.smoothing]).all() or args.gain <= 0 or args.speed <= 0 or args.smoothing < 0:
        parser.error("gain/speed must be finite and positive; smoothing finite and nonnegative")

    detector = HaarFaceMesh5pt(
        min_size=(48, 48),
        debug=False,
    )

    embedder = matcher = None
    if not args.any_face:
        embedder = ArcFaceEmbedderONNX(
            model_path="models/embedder_arcface.onnx",
            input_size=(112, 112),
            debug=False,
        )
        matcher = FaceDBMatcher(
            load_db_npz(Path("data/db/face_db.npz")),
            dist_thresh=args.threshold,
        )
        if args.target not in matcher.db:
            parser.error(f"Unknown identity {args.target!r}. Available: {', '.join(matcher.db)}")

    tracker = LockedFaceTracker(
        args.target,
        detector,
        embedder,
        matcher,
    )

    signals = FaceSignalExtractor()

    cap = cv2.VideoCapture(
        args.camera,
        cv2.CAP_DSHOW,
    )

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    window_name = "Locked Face Tracking"
    fullscreen = not args.windowed
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
    )

    blink_total = 0
    from src.hardware_tracking import AngleController
    from src.mqtt_tracking import MQTTLink
    controller = AngleController(args.mode, args.invert, args.min_angle,
                                 args.max_angle, gain=args.gain, speed=args.speed,
                                 smoothing=args.smoothing)
    link = None
    black_since = None
    black_warning_printed = False

    try:
        if not args.camera_only:
            link = MQTTLink(args.broker, args.mqtt_port)
            print("MQTT connected. Firmware starts from its own manual center at power-on.")
        print("Keys: Q quit | I invert | M pointer/pan | SPACE pause updates | R center.")
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            if frame is None or frame.size == 0:
                print("Camera returned an empty frame. Check Windows Camera and camera privacy settings.")
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            black_frame = int(frame.max()) == 0
            if black_frame:
                if black_since is None:
                    black_since = time.monotonic()
                if time.monotonic() - black_since >= 2 and not black_warning_printed:
                    print(f"CAMERA BLACK: camera {args.camera} returns all-zero pixels. Check the lens shutter, camera-disable key and Windows camera permissions. Test in Windows Camera, then close it before retrying.")
                    black_warning_printed = True
            else:
                black_since = None
                black_warning_printed = False
            locked_face, position = tracker.update(frame)

            # Existing tracker coasts visually on stale boxes; never drive the motor from those boxes.
            fresh = position is not None and tracker.miss_since is None
            if fresh:
                eye_mid = (locked_face.kps[0][0] + locked_face.kps[1][0]) / 2.0
                face_x = 0.6 * locked_face.kps[2][0] + 0.4 * eye_mid
                error = (face_x - frame.shape[1] / 2.0) / (frame.shape[1] / 2.0)
            else:
                error = None
            now = time.monotonic()
            angle = controller.update(error, now)
            if link:
                if fresh and not controller.paused:
                    link.send_angle(angle, now)
                link.poll()

            view = frame.copy()

            state_text = f"{tracker.state.name}: {args.target or 'FACE (no identity check)'}"

            state_color = (
                (0, 180, 0)
                if locked_face is not None
                else (0, 140, 255)
            )

            draw_label(
                view,
                state_text,
                (12, 28),
                state_color,
                0.72,
            )

            if locked_face is not None and position is not None:
                box = tracker.display_box()
                x1, y1, x2, y2 = box

                cv2.rectangle(
                    view,
                    (x1, y1),
                    (x2, y2),
                    (255, 170, 0),
                    3,
                )

                face_state = signals.analyze(
                    frame,
                    box,
                )

                if face_state is not None:
                    if face_state.blink:
                        blink_total += 1

                    expression = (
                        "SMILE"
                        if face_state.smiling
                        else "NEUTRAL"
                    )

                    eye_text = (
                        "EYES CLOSED"
                        if face_state.eyes_closed
                        else "EYES OPEN"
                    )

                    draw_label(
                        view,
                        expression,
                        (x1, max(55, y1 - 50)),
                        (0, 255, 255),
                    )

                    draw_label(
                        view,
                        f"{eye_text} blinks={blink_total}",
                        (x1, max(78, y1 - 25)),
                        (255, 255, 0),
                    )

                    draw_label(
                        view,
                        f"EAR={face_state.ear:.3f} "
                        f"smile={face_state.smile_score:.3f}",
                        (12, view.shape[0] - 18),
                        (255, 255, 255),
                        0.52,
                    )

                    draw_label(
                        view,
                        f"H={position.horizontal} "
                        f"V={position.vertical} "
                        f"error=({position.error_x:+.2f},"
                        f"{position.error_y:+.2f})",
                        (12, 56),
                        (255, 170, 0),
                        0.60,
                    )
            else:
                if tracker.state == LockState.SEARCHING:
                    signals.reset()

                h, w = view.shape[:2]
                dz = tracker.dead_zone

                cv2.rectangle(
                    view,
                    (
                        int(w * (0.5 - dz / 2)),
                        int(h * (0.5 - dz / 2)),
                    ),
                    (
                        int(w * (0.5 + dz / 2)),
                        int(h * (0.5 + dz / 2)),
                    ),
                    (120, 120, 120),
                    1,
                )

            if black_frame:
                draw_label(view, "CAMERA BLACK: check shutter / camera key / Windows Camera", (12, 115), (0, 100, 255), 0.48)
            hardware_text = "MQTT: " + link.poll() if link else "Camera only"
            draw_label(view, f"{args.mode} angle={angle} deg {'PAUSED' if controller.paused else ''}",
                       (12, 84), (255, 255, 0), 0.52)
            draw_label(view, hardware_text, (12, view.shape[0] - 42), (255, 255, 255), 0.45)
            cv2.imshow(window_name, view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("i"):
                controller.invert = not controller.invert
            elif key == ord("m"):
                args.mode = "pan" if args.mode == "pointer" else "pointer"
                controller.mode = args.mode
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
            elif key == ord(" "):
                controller.paused = not controller.paused
            elif key == ord("r"):
                controller.reset()
                if link:
                    link.send_angle(90, time.monotonic())

    finally:
        if link:
            try:
                link.close()
            except Exception as exc:
                print(f"MQTT shutdown: {exc}")
        cap.release()
        signals.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
