import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load models (cache to avoid reloading on every rerun)
@st.cache_resource
def load_lstm_model():
    return tf.keras.models.load_model('src/model/best_lstm_attention_autoencoder.keras')

@st.cache_resource
def load_gmm_model():
    return joblib.load('src/model/gmm_model.joblib')

model = load_lstm_model()
gmm = load_gmm_model()

st.title('LSTM Model Cluster Prediction')
st.write('Enter your input data (51 rows, 2 columns, comma-separated, one row per line):')

input_text = st.text_area('Input Data', height=200, placeholder='e.g.\n0.1,0.2\n0.3,0.4\n... (total 51 rows)')

if st.button('Predict'):
    try:
        rows = [list(map(float, row.split(','))) for row in input_text.strip().split('\n') if row.strip()]
        if len(rows) != 51 or any(len(r) != 2 for r in rows):
            st.error('Input must be 51 rows of 2 numbers each.')
        else:
            input_data = np.array(rows).reshape(1, 51, 2)
            encoder = model.get_layer('encoder') if 'encoder' in [l.name for l in model.layers] else model
            latent_vector = encoder.predict(input_data)
            cluster = int(gmm.predict(latent_vector)[0])
            st.success(f'Predicted cluster: {cluster}')
    except Exception as e:
        st.error(f'Error: {e}')
