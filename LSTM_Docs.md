# Documentation: LSTM + Attention for Gait Data Clustering

This notebook builds and trains an LSTM autoencoder with an attention mechanism to learn compressed (latent) representations of gait cycles, then clusters these representations to find patterns among patients.

---

## 1. Import Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
```
**Why?**  
- `pandas`, `numpy`: Data loading and manipulation.
- `matplotlib`: Plotting results.
- `tensorflow`, `keras`: Building and training neural networks.
- `sklearn`: Data splitting, scaling, and evaluation.

---

## 2. Define the LSTM Autoencoder with Attention

### a. Custom ReduceSum Layer

```python
from tensorflow.keras.layers import Layer

class ReduceSumLayer(Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)
```
**Why?**  
This custom layer sums across the time dimension, used to compute the context vector in the attention mechanism.

### b. LSTM + Attention Autoencoder Class

```python
class LSTMAttentionAutoencoder:
    def __init__(self, input_shape=(51, 2), latent_dim=16, lstm_units=[64, 32], dropout_rate=0.3, l2_reg=1e-4):
        # ... store parameters ...
        self.model, self.encoder = self._build_model()
    
    def _attention_block(self, lstm_out):
        # ... computes attention weights and context vector ...
        return context_vector

    def _build_model(self):
        # Encoder: LSTM layers + attention + dense layers to latent vector
        # Decoder: Dense + RepeatVector + LSTM layers to reconstruct input
        # Returns both autoencoder and encoder models
        return autoencoder, encoder

    def compile(self, learning_rate=1e-3):
        self.model.compile(optimizer=Adam(learning_rate), loss='mse')

    def train(self, X_train, epochs=50, batch_size=16, verbose=1):
        self.model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def get_latent_vectors(self, X):
        return self.encoder.predict(X)

    def summary(self):
        self.model.summary()
```
**Why?**  
- The class encapsulates the model architecture and training logic.
- The encoder compresses the input sequence into a latent vector.
- The decoder reconstructs the original sequence from the latent vector.
- The attention mechanism helps the model focus on important time steps.

---

## 3. Load and Prepare the Data

### a. Load Data

```python
data = pd.read_csv(r"C:\Users\Admin\Desktop\CP\Data\processed\final_processed\GAIT_analysis_dataset.csv")
```
**Why?**  
Loads the cleaned gait dataset.

### b. Split into Train and Test Sets by Patient

```python
unique_patients = data['Patient ID'].unique()
train_patients, test_patients = train_test_split(unique_patients, test_size=0.25, random_state=7)
train_df = data[data['Patient ID'].isin(train_patients)].reset_index(drop=True)
test_df = data[data['Patient ID'].isin(test_patients)].reset_index(drop=True)
```
**Why?**  
Ensures that all data from a patient is either in the train or test set, preventing data leakage.

### c. Fill Missing Values

```python
train_df = train_df.interpolate(axis=0).fillna(method='bfill').fillna(method='ffill')
test_df = test_df.interpolate(axis=0).fillna(method='bfill').fillna(method='ffill')
```
**Why?**  
Fills missing values using interpolation and forward/backward filling to ensure no NaNs remain.

### d. Extract Patient Sequences

```python
hip_cols = [col for col in train_df.columns if col.startswith('aSagH_')]
knee_cols = [col for col in train_df.columns if col.startswith('aSagK_')]

patient_sequences = []
patient_ids = []
for pid in train_df['Patient ID'].unique():
    hip_data = train_df[train_df['Patient ID'] == pid][hip_cols].values.flatten()
    knee_data = train_df[train_df['Patient ID'] == pid][knee_cols].values.flatten()
    if len(hip_data) == 51 and len(knee_data) == 51:
        patient_sequence = np.stack([knee_data, hip_data], axis=1)  # shape (51, 2)
        patient_sequences.append(patient_sequence)
        patient_ids.append(pid)
X_train = np.array(patient_sequences)
X_train_patient_ids = np.array(patient_ids)
```
**Why?**  
- For each patient, extracts their knee and hip angle sequences.
- Stacks them into a 2D array (frames, 2 features).
- Only includes patients with complete data (51 frames).

---

## 4. Scale the Data

```python
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_train = X_train_scaled.reshape(X_train.shape)
```
**Why?**  
- Scales all values to the [0, 1] range for better neural network training.
- Reshapes data for scaling, then restores original shape.

---

## 5. Hyperparameter Tuning (Optional)

```python
import keras_tuner as kt

def build_model(hp):
    # ... define model with tunable hyperparameters ...
    return model.model

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='...',
    project_name='lstm_attention_autoencoder'
)
tuner.search(X_train, X_train, epochs=30, batch_size=16, validation_split=0.2)
```
**Why?**  
Uses Keras Tuner to find the best model hyperparameters (e.g., latent dimension, LSTM units, dropout rate).

---

## 6. Train the Best Model

```python
best_model.save('best_lstm_attention_autoencoder.h5')
```
**Why?**  
Saves the best model for later use.

---

## 7. Evaluate Model: Reconstruction and Error Analysis

### a. Predict and Plot Reconstruction

```python
X_test_pred = best_model.model.predict(X_test)
plt.plot(actual[:, 0], label='Actual Knee')
plt.plot(reconstructed[:, 0], label='Reconstructed Knee', linestyle='--')
plt.plot(actual[:, 1], label='Actual Hip')
plt.plot(reconstructed[:, 1], label='Reconstructed Hip', linestyle='--')
plt.legend()
plt.show()
```
**Why?**  
Visualizes how well the model reconstructs the original gait cycles.

### b. Compute and Plot Errors

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse_list = [mean_squared_error(X_test[i], X_test_pred[i]) for i in range(len(X_test))]
mae_list = [mean_absolute_error(X_test[i], X_test_pred[i]) for i in range(len(X_test))]
plt.plot(mse_list, label='Test MSE per sample')
plt.plot(mae_list, label='Test MAE per sample')
plt.legend()
plt.show()
```
**Why?**  
Shows the reconstruction error for each test sample, helping to identify outliers or poor reconstructions.

---

## 8. Extract Latent Representations and Cluster

```python
X_train_latent = best_model.encoder.predict(X_train)
from sklearn.cluster import KMeans
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_train_latent)
```
**Why?**  
- The encoder compresses each patient's gait cycle into a latent vector.
- KMeans clusters these vectors to find groups of similar gait patterns.

---

## 9. Visualize Clusters in Latent Space

```python
from sklearn.decomposition import PCA
latent_2d = PCA(n_components=2).fit_transform(X_train_latent)
plt.scatter(latent_2d[:,0], latent_2d[:,1], c=clusters)
plt.title('Clustering of Latent Representations (Train Set)')
plt.show()
```
**Why?**  
Reduces the latent space to 2D for visualization, coloring points by cluster.

- Also uses t-SNE and UMAP for alternative visualizations.

---

## 10. Analyze and Plot Average Gait Cycles per Cluster

```python
for cluster_id in sorted(merged['Cluster'].unique()):
    cluster_data = merged[merged['Cluster'] == cluster_id]
    knee_mean = cluster_data[knee_cols].groupby(cluster_data['Patient ID']).mean().mean(axis=0)
    hip_mean = cluster_data[hip_cols].groupby(cluster_data['Patient ID']).mean().mean(axis=0)
    plt.plot(knee_mean, label='Knee')
    plt.plot(hip_mean, label='Hip')
    plt.title(f'Average Gait Cycle for Cluster {cluster_id}')
    plt.legend()
    plt.show()
```
**Why?**  
Shows the average knee and hip angle cycles for each cluster, helping interpret what each cluster represents.

---

## 11. Utility: Analyze Any Gait Dataset

```python
def analyze_gait_dataset(dataset_path, cluster_output_prefix):
    # Loads, processes, trains, clusters, and plots for any dataset
    # Saves cluster assignments and plots
```
**Why?**  
Reusable function to repeat the above workflow for any gait dataset.

---

## 12. Compare Cluster Assignments Across Datasets

```python
comparison = pd.merge(df_merged, df_original, on='Patient ID', suffixes=('_merged', '_original'))
diff = comparison[comparison['Cluster_merged'] != comparison['Cluster_original']]
```
**Why?**  
Checks if the same patient is assigned to different clusters in different datasets.

---

## 13. Count Patients per Cluster

```python
df['Cluster'].value_counts()
```
**Why?**  
Shows how many patients are in each cluster.

---

## Summary

- **Purpose:** Learn compressed representations of gait cycles using LSTM + attention, then cluster patients by gait pattern.
- **Key Steps:** Data loading, cleaning, sequence extraction, scaling, model building, training, clustering, and visualization.
- **Outcome:** Groups of patients with similar gait patterns, with visualizations and average cycles for interpretation.

---

**This workflow helps discover and interpret patterns in gait data, which can be useful for clinical research or patient stratification.**