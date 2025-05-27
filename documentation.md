Certainly! Here is a detailed, beginner-friendly documentation for your `data_cleaninig.ipynb` code, explaining each step, its purpose, and the logic behind it, with code snippets and plain English explanations.

---

# Documentation: Gait Data Cleaning and Preparation

This notebook prepares and cleans gait analysis data from multiple sources (CSV and Excel files), standardizes patient IDs, aligns columns, and saves the final dataset for further analysis.

---

## 1. Import Required Libraries

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
```
**Why?**  
- `pandas` is used for reading and manipulating tabular data.
- `numpy` is used for numerical operations and array handling.
- `MinMaxScaler` is used to normalize data values between 0 and 1.

---

## 2. Load the Main Dataset

```python
data = pd.read_csv(r'C:\Users\Admin\Desktop\CP\Data\processed\final_processed\GAIT_analysis_dataset.csv')
```
**Why?**  
Loads the main CSV file containing gait data for multiple patients.

---

## 3. Split and Standardize Patient IDs for Two Groups

```python
data_B = data[0:356]
data_B['Patient ID'] = data_B['Patient ID'].str.replace('^','DS_B_',regex=True)

data_A = data[356:712]
data_A['Patient ID'] = data_A['Patient ID'].str.replace('^CP','DS_A_',regex=True)
```
**Why?**  
- The dataset is split into two groups (A and B) based on row indices.
- Patient IDs are standardized by adding prefixes (`DS_B_` and `DS_A_`) for easier identification and to avoid duplicates.

---

## 4. Combine Groups A and B

```python
data_final = pd.concat([data_A, data_B], ignore_index=True)
```
**Why?**  
Combines the two groups into a single DataFrame for further processing.

---

## 5. Load and Clean Dataset C (Hip and Knee Data)

### a. Load Hip Data

```python
data_C_hip = pd.read_excel(r'C:\Users\Admin\Desktop\CP\Data\raw\Dataset_C.xlsx',sheet_name='bCP_Hip_sagittal')
```
**Why?**  
Loads hip angle data for group C from an Excel sheet.

### b. Rename Columns for Consistency

```python
import re

def rename_aSagH(col):
    match = re.match(r'aSagH_pct_GC_(\d+)', col)
    if match:
        num = int(match.group(1))
        return f'aSagH_{num//10}'
    return col

data_C_hip = data_C_hip.rename(columns=rename_aSagH)
```
**Why?**  
Renames columns from a verbose format (e.g., `aSagH_pct_GC_20`) to a simpler one (`aSagH_2`). This makes column names consistent and easier to work with.

### c. Keep Only Even-Indexed Columns

```python
cols_to_keep = ['Patient ID'] + [
    x for x in data_C_hip.columns
    if re.match(r'aSagH_(\d+)$', x) and int(re.match(r'aSagH_(\d+)$', x).group(1)) % 2 == 0
]
data_C_hip = data_C_hip[cols_to_keep]
```
**Why?**  
Keeps only columns where the index is even (e.g., `aSagH_0`, `aSagH_2`, ...), plus the patient ID. This reduces data size and ensures uniform sampling.

### d. Repeat for Knee Data

```python
data_C_knee = pd.read_excel(r'C:\Users\Admin\Desktop\CP\Data\raw\Dataset_C.xlsx',sheet_name='bCP_Knee_sagittal')

def rename_aSagK(col):
    match = re.match(r'aSagK_pct_GC_(\d+)', col)
    if match:
        num = int(match.group(1))
        return f'aSagK_{num//10}'
    return col

data_C_knee = data_C_knee.rename(columns=rename_aSagK)

cols_to_keep = ['Patient ID'] + [
    x for x in data_C_knee.columns
    if re.match(r'aSagK_(\d+)$', x) and int(re.match(r'aSagK_(\d+)$', x).group(1)) % 2 == 0
]
data_C_knee = data_C_knee[cols_to_keep]
```
**Why?**  
Same logic as for hip data, but for knee angles.

### e. Combine Hip and Knee Data for Group C

```python
x = list(data_C_knee.columns)
data_C = pd.concat([data_C_hip, data_C_knee[x[1:]]], axis=1)
data_C['Patient ID'] = data_C['Patient ID'].astype(str).str.replace('^','DS_C_',regex=True)
```
**Why?**  
- Combines hip and knee data for each patient.
- Standardizes patient IDs for group C.

---

## 6. Repeat for Dataset D

```python
data_D_hip = pd.read_excel(r"C:\Users\Admin\Desktop\CP\Data\raw\Dataset_D.xlsx",sheet_name='uCP_Hip_sagittal')
data_D_knee = pd.read_excel(r"C:\Users\Admin\Desktop\CP\Data\raw\Dataset_D.xlsx",sheet_name='uCP_Knee_sagittal')

data_D_hip = data_D_hip.rename(columns=rename_aSagH)
data_D_knee = data_D_knee.rename(columns=rename_aSagK)

cols_to_keep = ['Patient ID'] + [
    x for x in data_D_hip.columns
    if re.match(r'aSagH_(\d+)$', x) and int(re.match(r'aSagH_(\d+)$', x).group(1)) % 2 == 0
]
data_D_hip = data_D_hip[cols_to_keep]

cols_to_keep = [
    x for x in data_D_knee.columns
    if re.match(r'aSagK_(\d+)$', x) and int(re.match(r'aSagK_(\d+)$', x).group(1)) % 2 == 0
]
data_D_knee = data_D_knee[cols_to_keep]

data_D = pd.concat([data_D_hip, data_D_knee], axis=1)
data_D['Patient ID'] = data_D['Patient ID'].astype(str).str.replace('^','DS_D_',regex=True)
```
**Why?**  
- Loads, renames, filters, and combines hip and knee data for group D.
- Standardizes patient IDs for group D.

---

## 7. Combine All Groups into One Final Dataset

```python
final_data = pd.concat([data_final, data_C, data_D], ignore_index=True)
```
**Why?**  
Combines all cleaned and standardized data into a single DataFrame for further analysis.

---

## 8. Save the Final Dataset

```python
final_data.to_csv(r"C:\Users\Admin\Desktop\CP\Data\processed\final_processed\dataset_final.csv", index=False)
```
**Why?**  
Saves the cleaned, combined dataset to a CSV file for future use.

---

## 9. Prepare Data for Machine Learning

### a. Separate Hip and Knee Columns

```python
hip_cols = [col for col in df.columns if 'aSagH_' in col]
knee_cols = [col for col in df.columns if 'aSagK_' in col]
```
**Why?**  
Identifies which columns correspond to hip and knee angles.

### b. Stack Data for Each Patient

```python
hip_data = df[hip_cols].to_numpy()
knee_data = df[knee_cols].to_numpy()
data = np.stack([hip_data, knee_data], axis=-1)  # shape: (patients, frames, 2)
```
**Why?**  
Combines hip and knee data into a single 3D array for each patient, where the last dimension distinguishes between hip and knee.

### c. Normalize Data

```python
scalers = []
for i in range(data.shape[2]):
    scaler = MinMaxScaler()
    data[:, :, i] = scaler.fit_transform(data[:, :, i])
    scalers.append(scaler)
```
**Why?**  
Normalizes each feature (hip and knee) to a 0-1 range, which is important for many machine learning algorithms.

### d. Save as Numpy Array

```python
np.save("gait_data.npy", data)
```
**Why?**  
Saves the processed data in a format that is fast to load for machine learning tasks.

---

## Summary

- **Purpose:** Clean, standardize, and combine gait data from multiple sources for analysis.
- **Key Steps:** Load data, rename columns, filter relevant columns, standardize patient IDs, combine datasets, normalize, and save.
- **Outcome:** A single, clean, and ready-to-use dataset for further analysis or machine learning.

---

**This workflow ensures that all patient data is consistent, comparable, and ready for advanced analysis or modeling.**


