import os
import json
import tensorflow as tf

from data.dataloader import create_dataset
from models.xception_model import build_xception_model


EPOCHS = 10

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xception_deepfake.keras")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

os.makedirs(MODEL_DIR, exist_ok=True)


print("Loading datasets...")

train_dataset = create_dataset("train")
val_dataset = create_dataset("val")

print("Datasets loaded.")


print("\nBuilding Xception model...")

model = build_xception_model()

model.summary()


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    mode="max",
    verbose=1
)


print("\nStarting training...\n")


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


# Save final model
model.save(MODEL_PATH)


# Save training history
history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}


with open(HISTORY_PATH, "w") as file:
    json.dump(history_data, file, indent=4)


print("\nTraining complete!")

print(f"Model saved to: {MODEL_PATH}")
print(f"Training history saved to: {HISTORY_PATH}")
