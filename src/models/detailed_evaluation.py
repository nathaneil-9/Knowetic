import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

import matplotlib.pyplot as plt

from data.dataloader import create_dataset


MODEL_PATH = "models/xception_deepfake.keras"
OUTPUT_DIR = "docs/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("Loading test dataset...")
test_dataset = create_dataset("test")

print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

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
# Classification Report
# --------------------------------------------------

print("\nClassification Report:\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=["Real", "Fake"]
)

print(report)

with open(
    os.path.join(OUTPUT_DIR, "classification_report.txt"),
    "w"
) as f:
    f.write(report)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks([0, 1], ["Real", "Fake"])
plt.yticks([0, 1], ["Real", "Fake"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=300
)

plt.close()


# --------------------------------------------------
# ROC Curve + AUC
# --------------------------------------------------

auc_score = roc_auc_score(
    y_true,
    y_prob
)

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_prob
)

print(f"\nROC-AUC: {auc_score:.4f}")

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Xception (AUC = {auc_score:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "roc_curve.png"),
    dpi=300
)

plt.close()


print("\nDetailed evaluation complete!")

print(f"Results saved to: {OUTPUT_DIR}")
