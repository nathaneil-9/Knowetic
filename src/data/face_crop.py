import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

INPUT_DIRS = {
    "real": "data/processed/real",
    "fake": "data/processed/fake",
}

OUTPUT_DIRS = {
    "real": "data/processed/faces/real",
    "fake": "data/processed/faces/fake",
}

MODEL_PATH = "models/face_detector/blaze_face_short_range.tflite"

# Create MediaPipe face detector
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5,
)

detector = vision.FaceDetector.create_from_options(options)


def crop_face(image, detection):
    """Crop detected face with a small margin."""
    h, w = image.shape[:2]

    bbox = detection.bounding_box

    x = bbox.origin_x
    y = bbox.origin_y
    bw = bbox.width
    bh = bbox.height

    margin = 0.20

    x1 = max(0, int(x - bw * margin))
    y1 = max(0, int(y - bh * margin))
    x2 = min(w, int(x + bw * (1 + margin)))
    y2 = min(h, int(y + bh * (1 + margin)))

    face = image[y1:y2, x1:x2]

    if face.size == 0:
        return None

    return cv2.resize(face, (224, 224))


def process_dataset(label):
    input_dir = INPUT_DIRS[label]
    output_dir = OUTPUT_DIRS[label]

    os.makedirs(output_dir, exist_ok=True)

    total = 0
    saved = 0
    no_face = 0

    for root, _, files in os.walk(input_dir):
        for filename in files:

            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            input_path = os.path.join(root, filename)

            relative_path = os.path.relpath(root, input_dir)
            output_folder = os.path.join(output_dir, relative_path)

            os.makedirs(output_folder, exist_ok=True)

            output_path = os.path.join(output_folder, filename)

            total += 1

            image = cv2.imread(input_path)

            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            result = detector.detect(mp_image)

            if not result.detections:
                no_face += 1
                continue

            # Use the highest-confidence detection
            detection = max(
                result.detections,
                key=lambda d: d.categories[0].score
            )

            face = crop_face(image, detection)

            if face is None:
                no_face += 1
                continue

            cv2.imwrite(output_path, face)
            saved += 1

            if total % 500 == 0:
                print(
                    f"{label}: processed={total}, "
                    f"saved={saved}, no_face={no_face}"
                )

    print(f"\nFinished {label}")
    print(f"Total images: {total}")
    print(f"Faces saved: {saved}")
    print(f"No face detected: {no_face}")


if __name__ == "__main__":
    process_dataset("real")
    process_dataset("fake")

    detector.close()

    print("\nFace preprocessing complete.")
