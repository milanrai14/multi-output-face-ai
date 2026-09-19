import tensorflow as tf

MODEL_PATH = "model/best_model.keras"

model = None


def load_model():
    global model
    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded.")
    return model
