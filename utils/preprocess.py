# tb_detection_app/utils/preprocess.py
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize

def prepare_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

