import tensorflow as tf
import keras
print(dir(tf))  # Optional: to see what's inside tf

class CarPredictionModel:
    def __init__(self):
        self.model = tf.keras.models.load_model('temp_model.keras')

    def predict_car_type(self, image):

        self.model.predict()