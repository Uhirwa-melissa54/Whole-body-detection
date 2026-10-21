"""
Minimal pipeline:

camera -> Haar face box -> MediaPipe Face Landmarker
-> extract 5 keypoints -> draw

Run:

python -m src.landmarks

Keys:

q : quit
"""

import cv2
import numpy as np
import mediapipe as mp
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 5-point indices (Face Landmarker / FaceMesh topology)
IDX_LEFT_EYE = 33
IDX_RIGHT_EYE = 263
IDX_NOSE_TIP = 1
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291


def main():
    # Haar cascade
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    cascade_path = os.path.join(
        project_root,
        "data",
        "haarcascade_frontalface_default.xml",
    )

    if not os.path.exists(cascade_path):
        raise RuntimeError(
            f"Cascade file not found: {cascade_path}"
        )

    face = cv2.CascadeClassifier(cascade_path)

    if face.empty():
        raise RuntimeError(
            f"Failed to load cascade: {cascade_path}"
        )

    # MediaPipe Face Landmarker model
    model_path = os.path.join(
        project_root,
        "data",
        "face_landmarker.task",
    )

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Face Landmarker model not found: {model_path}"
        )

    # MediaPipe Face Landmarker
    base_options = python.BaseOptions(
        model_asset_path=model_path
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Camera not opened. Try camera index 0/1/2."
        )

    print(
        "Haar + MediaPipe Face Landmarker 5pt. "
        "Press 'q' to quit."
    )

    # Timestamp required by VIDEO mode.
    timestamp_ms = 0

    with vision.FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            H, W = frame.shape[:2]

            # -------------------------
            # Haar face detection
            # -------------------------
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = face.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )

            # Draw all Haar faces
            for (x, y, w, h) in faces:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

            # -------------------------
            # MediaPipe
            # -------------------------
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            timestamp_ms += 33

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            if result.face_landmarks:

                lm = result.face_landmarks[0]

                idxs = [
                    IDX_LEFT_EYE,
                    IDX_RIGHT_EYE,
                    IDX_NOSE_TIP,
                    IDX_MOUTH_LEFT,
                    IDX_MOUTH_RIGHT,
                ]

                pts = []

                for i in idxs:
                    p = lm[i]

                    pts.append([
                        p.x * W,
                        p.y * H,
                    ])

                kps = np.array(
                    pts,
                    dtype=np.float32,
                )

                # Enforce left/right ordering
                if kps[0, 0] > kps[1, 0]:
                    kps[[0, 1]] = kps[[1, 0]]

                if kps[3, 0] > kps[4, 0]:
                    kps[[3, 4]] = kps[[4, 3]]

                # Draw 5 points
                for (px, py) in kps.astype(int):

                    cv2.circle(
                        frame,
                        (int(px), int(py)),
                        4,
                        (0, 255, 0),
                        -1,
                    )

                cv2.putText(
                    frame,
                    "5pt",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(
                "5pt Landmarks",
                frame,
            )

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()