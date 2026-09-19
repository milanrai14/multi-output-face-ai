import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

from model_loader import load_model

IMG_SIZE = (224, 224)


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_age_gender(image: Image.Image):
    model = load_model()

    image_array = preprocess_image(image)

    age_pred, gender_pred = model.predict(image_array)

    age = round(float(age_pred[0][0]))
    gender_probability = float(gender_pred[0][0])
    gender = "Male" if gender_probability < 0.5 else "Female"

    return {
        "age": age,
        "gender": gender,
        "gender_probability": gender_probability,
    }
