# src/detect.py

import cv2
import os


def main():
    cascade_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "haarcascade_frontalface_default.xml",
)
    if not os.path.exists(cascade_path):
     raise RuntimeError(f"Cascade file not found at: {cascade_path}")

    face = cv2.CascadeClassifier(cascade_path)

    if face.empty():
        raise RuntimeError(f"Failed to load cascade: {cascade_path}")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Try camera index 0/1/2.")

    print("Haar face detect (minimal). Press 'q' to quit.")

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Minimal but practical defaults
        faces = face.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

        cv2.imshow("Face Detection", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()