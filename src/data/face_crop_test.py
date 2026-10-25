import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

MODEL_PATH = "models/face_detector/blaze_face_short_range.tflite"
INPUT_DIR = Path("data/processed/real")
OUTPUT_DIR = Path("data/processed/faces_test")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5,
)

detector = vision.FaceDetector.create_from_options(options)

files = list(INPUT_DIR.rglob("*.jpg"))[:5]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for image_path in files:
    image = cv2.imread(str(image_path))

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    result = detector.detect(mp_image)

    if not result.detections:
        print("No face:", image_path)
        continue

    detection = max(
        result.detections,
        key=lambda d: d.categories[0].score
    )

    bbox = detection.bounding_box

    x = max(0, bbox.origin_x)
    y = max(0, bbox.origin_y)
    x2 = min(image.shape[1], x + bbox.width)
    y2 = min(image.shape[0], y + bbox.height)

    face = image[y:y2, x:x2]

    if face.size == 0:
        print("Invalid crop:", image_path)
        continue

    face = cv2.resize(face, (224, 224))

    output_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(output_path), face)

    print("Saved:", output_path)

detector.close()

print("Test complete.")
