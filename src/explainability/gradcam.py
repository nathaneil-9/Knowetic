import tensorflow as tf
import numpy as np
import cv2
import os


MODEL_PATH = "models/xception_deepfake.keras"
IMAGE_SIZE = (224, 224)
LAST_CONV_LAYER = "block14_sepconv2_act"

IMAGE_PATH = "data/processed/split/test/fake/575_603/frame_07.jpg"
OUTPUT_PATH = "data/processed/gradcam_result.jpg"


def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original = image.copy()

    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype(np.float32)

    image = tf.keras.applications.xception.preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image, original


def make_gradcam(model, image, last_conv_layer_name):
    # Get the Xception base model
    base_model = model.get_layer("xception")

    # Model for Grad-CAM
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(last_conv_layer_name).output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:

        # Get convolutional feature maps
        conv_outputs, base_output = grad_model(image)

        # Pass through the classifier
        x = model.layers[1](base_output)
        x = model.layers[2](x, training=False)
        x = model.layers[3](x)
        x = model.layers[4](x, training=False)
        predictions = model.layers[5](x)

        fake_probability = predictions[:, 0]

        # Explain the class that the model actually predicted
        target = tf.where(
            fake_probability >= 0.5,
            fake_probability,
            1.0 - fake_probability
        )

    # Calculate gradients for the predicted class
    gradients = tape.gradient(
        target,
        conv_outputs
    )

    # Average gradients across spatial dimensions
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    # Weight feature maps by their importance
    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    # Keep positive activations
    heatmap = tf.maximum(heatmap, 0)

    # Normalize
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy(), float(fake_probability[0])

def create_overlay(original, heatmap):

    # Resize heatmap to original image dimensions
    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0])
    )

    # Convert heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)

    # Apply color map
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Convert original image back to BGR
    original_bgr = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    # Combine original image and heatmap
    overlay = cv2.addWeighted(
        original_bgr,
        0.6,
        heatmap,
        0.4,
        0
    )

    return overlay


if __name__ == "__main__":

    print("Loading model...")
    model = load_model()

    print("Loading image...")
    image, original = preprocess_image(IMAGE_PATH)

    print("Generating Grad-CAM...")

    heatmap, prediction = make_gradcam(
        model,
        image,
        LAST_CONV_LAYER
    )

    # Convert prediction to Real/Fake
    if prediction >= 0.5:
        result = "Fake"
        confidence = prediction
    else:
        result = "Real"
        confidence = 1 - prediction

    print(f"\nPrediction: {result}")
    print(f"Confidence: {confidence * 100:.2f}%")

    # Create heatmap overlay
    overlay = create_overlay(
        original,
        heatmap
    )

    # Make sure output directory exists
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    # Save result
    cv2.imwrite(
        OUTPUT_PATH,
        overlay
    )

    print(f"Grad-CAM saved to: {OUTPUT_PATH}")
