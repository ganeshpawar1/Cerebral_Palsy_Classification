from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load the saved LSTM model
model = tf.keras.models.load_model('model/best_lstm_attention_autoencoder.keras')

# Load the saved GMM model
gmm = joblib.load('model/gmm_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape(1, 51, 2)  # Reshape input for the model

    # Get the latent vector from the encoder
    encoder = model.get_layer('encoder') if 'encoder' in [l.name for l in model.layers] else model
    latent_vector = encoder.predict(input_data)

    # Predict the cluster using the GMM model
    cluster = int(gmm.predict(latent_vector)[0])

    return jsonify({'cluster': cluster})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)