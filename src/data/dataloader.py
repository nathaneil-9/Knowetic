import tensorflow as tf

# Dataset settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 42

DATA_DIR = "data/processed/split"


def create_dataset(split):
    """
    Create a TensorFlow dataset for the given split.

    Labels:
        real = 0
        fake = 1
    """

    directory = f"{DATA_DIR}/{split}"

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
	class_names=["real", "fake"],
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        seed=SEED,
    )

    # Normalize pixel values from [0, 255] to [0, 1]
    preprocessing = tf.keras.applications.xception.preprocess_input

    dataset = dataset.map(
        lambda images, labels: (
            preprocessing(images),
            labels
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


if __name__ == "__main__":

    train_dataset = create_dataset("train")
    val_dataset = create_dataset("val")
    test_dataset = create_dataset("test")

    print("\nDatasets loaded successfully!")

    print("Train batches:", tf.data.experimental.cardinality(train_dataset).numpy())
    print("Validation batches:", tf.data.experimental.cardinality(val_dataset).numpy())
    print("Test batches:", tf.data.experimental.cardinality(test_dataset).numpy())

