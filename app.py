import numpy as np
from keras.models import load_model
import imageio
from flask import Flask, request, jsonify, render_template

# Cargar el modelo entrenado
model = load_model('mnist_cnn_model.h5')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener la imagen desde la solicitud POST
    file = request.files['image']
    image = imageio.imread(file)
    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])

    # Asegurarnos de que la imagen tenga la forma correcta
    if gray.shape != (28, 28):
        gray = gray[:28, :28]  # Ajustar la forma a 28x28

    # Preprocesar la imagen para que coincida con el formato de entrada del modelo
    gray = gray.reshape(1, 28, 28, 1)
    gray = gray.astype('float32') / 255

    # Realizar la predicción
    prediction = model.predict(gray)
    predicted_number = prediction.argmax()

    return jsonify({'predicted_number': int(predicted_number)})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
