# tb_detection_app/utils/gradcam.py
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model
from matplotlib import cm
from PIL import Image

def generate_gradcam(model, img_array, last_conv_layer_name):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Convert to RGB heatmap
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[np.uint8(255 * heatmap)]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[2], img_array.shape[1]))
    jet_heatmap = np.array(jet_heatmap)

    # Superimpose on original image
    original = tf.keras.utils.array_to_img(img_array[0])
    original = np.array(original)
    superimposed_img = jet_heatmap * 0.4 + original
    superimposed_img = np.uint8(superimposed_img)
    return Image.fromarray(superimposed_img)

