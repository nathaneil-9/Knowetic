import os
import numpy as np
import tensorflow as tf
import cv2

from data.dataloader import create_dataset


MODEL_PATH = "models/xception_deepfake.keras"
DATA_DIR = "data/processed/split/test"
OUTPUT_DIR = "docs/evaluation/error_analysis"

MAX_EXAMPLES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("Loading test dataset...")
test_dataset = create_dataset("test")

print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)


# Get file paths in the same order as the dataset
file_paths = []

for class_name in ["real", "fake"]:
    class_dir = os.path.join(DATA_DIR, class_name)

    for root, _, files in os.walk(class_dir):
        for file in sorted(files):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                file_paths.append(
                    os.path.join(root, file)
                )

file_paths = sorted(file_paths)


print("\nGenerating predictions...\n")

y_true = []
y_prob = []


for images, labels in test_dataset:
    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy().flatten())
    y_prob.extend(predictions.flatten())


y_true = np.array(y_true).astype(int)
y_prob = np.array(y_prob)
y_pred = (y_prob >= 0.5).astype(int)


# --------------------------------------------------
# Find errors
# --------------------------------------------------

false_positives = np.where(
    (y_true == 0) & (y_pred == 1)
)[0]

false_negatives = np.where(
    (y_true == 1) & (y_pred == 0)
)[0]


print(f"False Positives: {len(false_positives)}")
print(f"False Negatives: {len(false_negatives)}")


# --------------------------------------------------
# Save representative examples
# --------------------------------------------------

fp_dir = os.path.join(
    OUTPUT_DIR,
    "false_positives"
)

fn_dir = os.path.join(
    OUTPUT_DIR,
    "false_negatives"
)

os.makedirs(fp_dir, exist_ok=True)
os.makedirs(fn_dir, exist_ok=True)


def save_examples(indices, output_dir, error_type):

    # Select the most confident errors first
    if error_type == "false_positive":
        confidence = y_prob[indices]
        sorted_indices = indices[np.argsort(-confidence)]
    else:
        confidence = 1 - y_prob[indices]
        sorted_indices = indices[np.argsort(-confidence)]

    selected = sorted_indices[:MAX_EXAMPLES]

    for number, index in enumerate(selected):

        image_path = file_paths[index]

        image = cv2.imread(image_path)

        if image is None:
            continue

        probability = y_prob[index]

        if probability >= 0.5:
            prediction = "Fake"
            confidence_value = probability
        else:
            prediction = "Real"
            confidence_value = 1 - probability

        filename = (
            f"{number + 1:02d}_"
            f"predicted_{prediction}_"
            f"confidence_{confidence_value:.2f}.jpg"
        )

        output_path = os.path.join(
            output_dir,
            filename
        )

        cv2.imwrite(
            output_path,
            image
        )


save_examples(
    false_positives,
    fp_dir,
    "false_positive"
)

save_examples(
    false_negatives,
    fn_dir,
    "false_negative"
)


# --------------------------------------------------
# Save summary
# --------------------------------------------------

summary_path = os.path.join(
    OUTPUT_DIR,
    "error_analysis_summary.txt"
)


with open(summary_path, "w") as file:

    file.write("Knowetic - Model Error Analysis\n")
    file.write("=" * 40 + "\n\n")

    file.write(
        f"Total test images: {len(y_true)}\n"
    )

    file.write(
        f"False positives: {len(false_positives)}\n"
    )

    file.write(
        f"False negatives: {len(false_negatives)}\n"
    )

    file.write(
        f"False positive rate: "
        f"{len(false_positives) / np.sum(y_true == 0):.4f}\n"
    )

    file.write(
        f"False negative rate: "
        f"{len(false_negatives) / np.sum(y_true == 1):.4f}\n"
    )

    file.write("\n")
    file.write(
        "Representative error images were saved in:\n"
    )

    file.write(
        f"- {fp_dir}\n"
    )

    file.write(
        f"- {fn_dir}\n"
    )


print("\nError analysis complete!")

print(f"Results saved to: {OUTPUT_DIR}")
