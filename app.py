import os
import numpy as np
from flask import Flask, request, render_template
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('models/your_trained_model.h5')
class_labels = ["Bugatti", "Nissan_GTR", "Rolls_Royce","Supra","ferrari","lamborghini - urus"]

@app.route('/')     
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        image = Image.open(file)
        image = image.resize((224, 224))  # Resize to match model input size
        image = np.array(image) / 255.0  # Normalize

        predictions = model.predict(np.expand_dims(image, axis=0))
        predicted_label = class_labels[np.argmax(predictions)]

        return f"Predicted car model: {predicted_label}"
#ctrl+shift+p
if __name__ == '__main__':
    app.run(debug=True)
