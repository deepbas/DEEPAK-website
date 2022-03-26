---
authors:
- admin
categories: [machine learning, data science, prediction]
date: "2022-03-23T00:00:00Z"
image:
  caption: ""
  focal_point: ""
lastMod: "2022-03-23T00:00:00Z"
projects: []
subtitle: Learn how to make a research blog with hugo blogdown
summary: Learn more about Python and Jupyter labs
tags: []
title: How to predict covid case counts using machine learning models?
---

```python
from IPython.core.display import Image
Image('https://www.python.org/static/community_logos/python-logo-master-v3-TM-flattened.png')
```

![png](./index_1_0.png)---
title: "Predicting covid cases with LSTM Machine Learning Model"
date: 2020-03-20
tags: ["data science", "machine learning", "hugo"]
draft: false
---


```python
# Import various libraries and routines needed for computation
import math 
import pandas as pd
import numpy as np
import tensorflow as tf
%matplotlib inline
import matplotlib.pyplot as plt
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.backend as K
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from datetime import date, timedelta, datetime 
```


```python
df = pd.read_csv('covid_final.csv')  
dataset = df.set_index(['date'])
dataset.drop(dataset.tail(10).index,
        inplace = True)
values = dataset.values
```


```python
date_index = dataset.index
```


```python
data_clean = dataset.copy()
data_clean_ext = dataset.copy()
data_clean_ext['new_cases_predictions'] = data_clean_ext['new_cases_smoothed']
data_clean.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>new_cases_smoothed</th>
      <th>reproduction_rate</th>
      <th>new_tests_smoothed_per_thousand</th>
      <th>new_vaccinations_smoothed_per_million</th>
      <th>people_fully_vaccinated_per_hundred</th>
      <th>total_boosters_per_hundred</th>
      <th>stringency_index</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-03-08</th>
      <td>38934.286</td>
      <td>0.65</td>
      <td>2.748</td>
      <td>621</td>
      <td>65.24</td>
      <td>28.89</td>
      <td>53.24</td>
    </tr>
    <tr>
      <th>2022-03-09</th>
      <td>36641.429</td>
      <td>0.66</td>
      <td>2.699</td>
      <td>601</td>
      <td>65.25</td>
      <td>28.91</td>
      <td>53.24</td>
    </tr>
    <tr>
      <th>2022-03-10</th>
      <td>36330.429</td>
      <td>0.69</td>
      <td>2.613</td>
      <td>583</td>
      <td>65.27</td>
      <td>28.94</td>
      <td>53.24</td>
    </tr>
    <tr>
      <th>2022-03-11</th>
      <td>36104.714</td>
      <td>0.71</td>
      <td>2.580</td>
      <td>557</td>
      <td>65.29</td>
      <td>28.97</td>
      <td>53.24</td>
    </tr>
    <tr>
      <th>2022-03-12</th>
      <td>35464.143</td>
      <td>0.71</td>
      <td>2.561</td>
      <td>540</td>
      <td>65.30</td>
      <td>28.99</td>
      <td>53.24</td>
    </tr>
  </tbody>
</table>
</div>




```python
# number of rows in the data
nrows = data_clean.shape[0]
```


```python
# Convert the data to numpy values
np_data_unscaled = np.array(data_clean)
np_data = np.reshape(np_data_unscaled, (nrows, -1))
```


```python
# ensure all data is float
values = values.astype('float64')
```


```python
# Transform the data by scaling each feature to a range between 0 and 1
scaler = MinMaxScaler()
np_data_scaled = scaler.fit_transform(np_data_unscaled)
```


```python
# Creating a separate scaler that works on a single column for scaling predictions
scaler_pred = MinMaxScaler()
df_cases = pd.DataFrame(data_clean_ext['new_cases_smoothed'])
np_cases_scaled = scaler_pred.fit_transform(df_cases)
```


```python
# Set the sequence length - this is the timeframe used to make a single prediction
sequence_length = 31

# Prediction Index
index_cases = dataset.columns.get_loc("new_cases_smoothed")

# Split the training data into train and train data sets
# As a first step, we get the number of rows to train the model on 80% of the data 
train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)

# Create the training and test data
train_data = np_data_scaled[0:train_data_len, :]
test_data = np_data_scaled[train_data_len - sequence_length:, :]

# The RNN needs data with the format of [samples, time steps, features]
# Here, we create N samples, sequence_length time steps per sample, and 6 features
def partition_dataset(sequence_length, data):
    x, y = [], []
    data_len = data.shape[0]
    for i in range(sequence_length, data_len):
        x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
        y.append(data[i, index_cases]) #contains the prediction values for validation,  for single-step prediction
    
    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y

# Generate training data and test data
x_train, y_train = partition_dataset(sequence_length, train_data)
x_test, y_test = partition_dataset(sequence_length, test_data)
```


```python

# Configure the neural network model
model = Sequential()
# Model with n_neurons = inputshape Timestamps, each with x_train.shape[2] variables
n_neurons = x_train.shape[1] * x_train.shape[2]
model.add(LSTM(n_neurons, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(1))
```


```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# Compiling the LSTM
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
```


```python
checkpoint_path = 'my_best_model.hdf5'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')

earlystopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose =0)
callbacks = [checkpoint, earlystopping]
```


```python
# Training the model
epochs = 300
batch_size = 20
history = model.fit(x_train, y_train,
                     batch_size=batch_size, 
                     epochs=epochs,
                     validation_data=(x_test, y_test),
                     callbacks = callbacks,
                     verbose = 0)
```

    
    Epoch 00001: val_loss improved from inf to 0.03988, saving model to my_best_model.hdf5
    
    Epoch 00002: val_loss improved from 0.03988 to 0.03568, saving model to my_best_model.hdf5
    
    Epoch 00003: val_loss did not improve from 0.03568
    
    Epoch 00004: val_loss did not improve from 0.03568
    
    Epoch 00005: val_loss did not improve from 0.03568
    
    Epoch 00006: val_loss did not improve from 0.03568
    
    Epoch 00007: val_loss did not improve from 0.03568
    
    Epoch 00008: val_loss did not improve from 0.03568
    
    Epoch 00009: val_loss did not improve from 0.03568
    
    Epoch 00010: val_loss did not improve from 0.03568
    
    Epoch 00011: val_loss did not improve from 0.03568
    
    Epoch 00012: val_loss did not improve from 0.03568
    
    Epoch 00013: val_loss did not improve from 0.03568
    
    Epoch 00014: val_loss did not improve from 0.03568
    
    Epoch 00015: val_loss did not improve from 0.03568
    
    Epoch 00016: val_loss did not improve from 0.03568
    
    Epoch 00017: val_loss did not improve from 0.03568
    
    Epoch 00018: val_loss did not improve from 0.03568
    
    Epoch 00019: val_loss did not improve from 0.03568
    
    Epoch 00020: val_loss did not improve from 0.03568
    
    Epoch 00021: val_loss did not improve from 0.03568
    
    Epoch 00022: val_loss did not improve from 0.03568
    
    Epoch 00023: val_loss did not improve from 0.03568
    
    Epoch 00024: val_loss did not improve from 0.03568
    
    Epoch 00025: val_loss did not improve from 0.03568
    
    Epoch 00026: val_loss did not improve from 0.03568
    
    Epoch 00027: val_loss did not improve from 0.03568
    
    Epoch 00028: val_loss did not improve from 0.03568
    
    Epoch 00029: val_loss did not improve from 0.03568
    
    Epoch 00030: val_loss did not improve from 0.03568
    
    Epoch 00031: val_loss did not improve from 0.03568
    
    Epoch 00032: val_loss did not improve from 0.03568
    
    Epoch 00033: val_loss did not improve from 0.03568
    
    Epoch 00034: val_loss did not improve from 0.03568
    
    Epoch 00035: val_loss did not improve from 0.03568
    
    Epoch 00036: val_loss did not improve from 0.03568
    
    Epoch 00037: val_loss did not improve from 0.03568
    
    Epoch 00038: val_loss did not improve from 0.03568
    
    Epoch 00039: val_loss did not improve from 0.03568
    
    Epoch 00040: val_loss did not improve from 0.03568
    
    Epoch 00041: val_loss improved from 0.03568 to 0.03540, saving model to my_best_model.hdf5
    
    Epoch 00042: val_loss improved from 0.03540 to 0.03177, saving model to my_best_model.hdf5
    
    Epoch 00043: val_loss improved from 0.03177 to 0.02654, saving model to my_best_model.hdf5
    
    Epoch 00044: val_loss did not improve from 0.02654
    
    Epoch 00045: val_loss did not improve from 0.02654
    
    Epoch 00046: val_loss improved from 0.02654 to 0.02444, saving model to my_best_model.hdf5
    
    Epoch 00047: val_loss did not improve from 0.02444
    
    Epoch 00048: val_loss improved from 0.02444 to 0.02441, saving model to my_best_model.hdf5
    
    Epoch 00049: val_loss improved from 0.02441 to 0.02097, saving model to my_best_model.hdf5
    
    Epoch 00050: val_loss did not improve from 0.02097
    
    Epoch 00051: val_loss did not improve from 0.02097
    
    Epoch 00052: val_loss did not improve from 0.02097
    
    Epoch 00053: val_loss did not improve from 0.02097
    
    Epoch 00054: val_loss did not improve from 0.02097
    
    Epoch 00055: val_loss did not improve from 0.02097
    
    Epoch 00056: val_loss improved from 0.02097 to 0.02015, saving model to my_best_model.hdf5
    
    Epoch 00057: val_loss did not improve from 0.02015
    
    Epoch 00058: val_loss did not improve from 0.02015
    
    Epoch 00059: val_loss did not improve from 0.02015
    
    Epoch 00060: val_loss improved from 0.02015 to 0.01943, saving model to my_best_model.hdf5
    
    Epoch 00061: val_loss improved from 0.01943 to 0.01820, saving model to my_best_model.hdf5
    
    Epoch 00062: val_loss improved from 0.01820 to 0.01687, saving model to my_best_model.hdf5
    
    Epoch 00063: val_loss improved from 0.01687 to 0.01529, saving model to my_best_model.hdf5
    
    Epoch 00064: val_loss did not improve from 0.01529
    
    Epoch 00065: val_loss did not improve from 0.01529
    
    Epoch 00066: val_loss did not improve from 0.01529
    
    Epoch 00067: val_loss did not improve from 0.01529
    
    Epoch 00068: val_loss did not improve from 0.01529
    
    Epoch 00069: val_loss did not improve from 0.01529
    
    Epoch 00070: val_loss improved from 0.01529 to 0.01523, saving model to my_best_model.hdf5
    
    Epoch 00071: val_loss improved from 0.01523 to 0.01438, saving model to my_best_model.hdf5
    
    Epoch 00072: val_loss improved from 0.01438 to 0.01253, saving model to my_best_model.hdf5
    
    Epoch 00073: val_loss did not improve from 0.01253
    
    Epoch 00074: val_loss did not improve from 0.01253
    
    Epoch 00075: val_loss did not improve from 0.01253
    
    Epoch 00076: val_loss did not improve from 0.01253
    
    Epoch 00077: val_loss did not improve from 0.01253
    
    Epoch 00078: val_loss did not improve from 0.01253
    
    Epoch 00079: val_loss improved from 0.01253 to 0.00920, saving model to my_best_model.hdf5
    
    Epoch 00080: val_loss did not improve from 0.00920
    
    Epoch 00081: val_loss did not improve from 0.00920
    
    Epoch 00082: val_loss did not improve from 0.00920
    
    Epoch 00083: val_loss did not improve from 0.00920
    
    Epoch 00084: val_loss did not improve from 0.00920
    
    Epoch 00085: val_loss did not improve from 0.00920
    
    Epoch 00086: val_loss did not improve from 0.00920
    
    Epoch 00087: val_loss did not improve from 0.00920
    
    Epoch 00088: val_loss did not improve from 0.00920
    
    Epoch 00089: val_loss improved from 0.00920 to 0.00801, saving model to my_best_model.hdf5
    
    Epoch 00090: val_loss did not improve from 0.00801
    
    Epoch 00091: val_loss did not improve from 0.00801
    
    Epoch 00092: val_loss did not improve from 0.00801
    
    Epoch 00093: val_loss did not improve from 0.00801
    
    Epoch 00094: val_loss did not improve from 0.00801
    
    Epoch 00095: val_loss did not improve from 0.00801
    
    Epoch 00096: val_loss did not improve from 0.00801
    
    Epoch 00097: val_loss did not improve from 0.00801
    
    Epoch 00098: val_loss did not improve from 0.00801
    
    Epoch 00099: val_loss did not improve from 0.00801
    
    Epoch 00100: val_loss did not improve from 0.00801
    
    Epoch 00101: val_loss did not improve from 0.00801
    
    Epoch 00102: val_loss did not improve from 0.00801
    
    Epoch 00103: val_loss improved from 0.00801 to 0.00780, saving model to my_best_model.hdf5
    
    Epoch 00104: val_loss did not improve from 0.00780
    
    Epoch 00105: val_loss did not improve from 0.00780
    
    Epoch 00106: val_loss did not improve from 0.00780
    
    Epoch 00107: val_loss did not improve from 0.00780
    
    Epoch 00108: val_loss did not improve from 0.00780
    
    Epoch 00109: val_loss did not improve from 0.00780
    
    Epoch 00110: val_loss improved from 0.00780 to 0.00678, saving model to my_best_model.hdf5
    
    Epoch 00111: val_loss did not improve from 0.00678
    
    Epoch 00112: val_loss did not improve from 0.00678
    
    Epoch 00113: val_loss did not improve from 0.00678
    
    Epoch 00114: val_loss did not improve from 0.00678
    
    Epoch 00115: val_loss did not improve from 0.00678
    
    Epoch 00116: val_loss did not improve from 0.00678
    
    Epoch 00117: val_loss did not improve from 0.00678
    
    Epoch 00118: val_loss improved from 0.00678 to 0.00657, saving model to my_best_model.hdf5
    
    Epoch 00119: val_loss did not improve from 0.00657
    
    Epoch 00120: val_loss did not improve from 0.00657
    
    Epoch 00121: val_loss did not improve from 0.00657
    
    Epoch 00122: val_loss did not improve from 0.00657
    
    Epoch 00123: val_loss did not improve from 0.00657
    
    Epoch 00124: val_loss did not improve from 0.00657
    
    Epoch 00125: val_loss improved from 0.00657 to 0.00530, saving model to my_best_model.hdf5
    
    Epoch 00126: val_loss did not improve from 0.00530
    
    Epoch 00127: val_loss did not improve from 0.00530
    
    Epoch 00128: val_loss did not improve from 0.00530
    
    Epoch 00129: val_loss did not improve from 0.00530
    
    Epoch 00130: val_loss did not improve from 0.00530
    
    Epoch 00131: val_loss did not improve from 0.00530
    
    Epoch 00132: val_loss did not improve from 0.00530
    
    Epoch 00133: val_loss did not improve from 0.00530
    
    Epoch 00134: val_loss did not improve from 0.00530
    
    Epoch 00135: val_loss did not improve from 0.00530
    
    Epoch 00136: val_loss did not improve from 0.00530
    
    Epoch 00137: val_loss did not improve from 0.00530
    
    Epoch 00138: val_loss did not improve from 0.00530
    
    Epoch 00139: val_loss did not improve from 0.00530
    
    Epoch 00140: val_loss did not improve from 0.00530
    
    Epoch 00141: val_loss did not improve from 0.00530
    
    Epoch 00142: val_loss did not improve from 0.00530
    
    Epoch 00143: val_loss did not improve from 0.00530
    
    Epoch 00144: val_loss improved from 0.00530 to 0.00444, saving model to my_best_model.hdf5
    
    Epoch 00145: val_loss did not improve from 0.00444
    
    Epoch 00146: val_loss did not improve from 0.00444
    
    Epoch 00147: val_loss did not improve from 0.00444
    
    Epoch 00148: val_loss did not improve from 0.00444
    
    Epoch 00149: val_loss did not improve from 0.00444
    
    Epoch 00150: val_loss did not improve from 0.00444
    
    Epoch 00151: val_loss did not improve from 0.00444
    
    Epoch 00152: val_loss did not improve from 0.00444
    
    Epoch 00153: val_loss did not improve from 0.00444
    
    Epoch 00154: val_loss did not improve from 0.00444
    
    Epoch 00155: val_loss did not improve from 0.00444
    
    Epoch 00156: val_loss did not improve from 0.00444
    
    Epoch 00157: val_loss did not improve from 0.00444
    
    Epoch 00158: val_loss did not improve from 0.00444
    
    Epoch 00159: val_loss did not improve from 0.00444
    
    Epoch 00160: val_loss did not improve from 0.00444
    
    Epoch 00161: val_loss improved from 0.00444 to 0.00436, saving model to my_best_model.hdf5
    
    Epoch 00162: val_loss did not improve from 0.00436
    
    Epoch 00163: val_loss did not improve from 0.00436
    
    Epoch 00164: val_loss did not improve from 0.00436
    
    Epoch 00165: val_loss improved from 0.00436 to 0.00359, saving model to my_best_model.hdf5
    
    Epoch 00166: val_loss did not improve from 0.00359
    
    Epoch 00167: val_loss did not improve from 0.00359
    
    Epoch 00168: val_loss did not improve from 0.00359
    
    Epoch 00169: val_loss did not improve from 0.00359
    
    Epoch 00170: val_loss did not improve from 0.00359
    
    Epoch 00171: val_loss did not improve from 0.00359
    
    Epoch 00172: val_loss did not improve from 0.00359
    
    Epoch 00173: val_loss did not improve from 0.00359
    
    Epoch 00174: val_loss did not improve from 0.00359
    
    Epoch 00175: val_loss improved from 0.00359 to 0.00298, saving model to my_best_model.hdf5
    
    Epoch 00176: val_loss did not improve from 0.00298
    
    Epoch 00177: val_loss did not improve from 0.00298
    
    Epoch 00178: val_loss did not improve from 0.00298
    
    Epoch 00179: val_loss did not improve from 0.00298
    
    Epoch 00180: val_loss did not improve from 0.00298



```python
from tensorflow.keras.models import load_model
model_from_saved_checkpoint = load_model(checkpoint_path)
```


```python
# Plot training & validation loss values
plt.figure(figsize=(16,7))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
```


```python
# Get the predicted values
y_pred_scaled = model_from_saved_checkpoint.predict(x_test)
```


```python
# Unscale the predicted values
y_pred = scaler_pred.inverse_transform(y_pred_scaled)
```


```python
y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))
```


```python
# Mean Absolute Error (MAE)
MAE = mean_absolute_error(y_test_unscaled, y_pred)
print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')

# Mean Absolute Percentage Error (MAPE)
MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

# Median Absolute Percentage Error (MDAPE)
MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled)) ) * 100
print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')
```


```python
plt.plot(y_test_unscaled, label='True')
plt.plot(y_pred, label='LSTM')
plt.title("LSTM's_Prediction")
plt.xlabel('Observation')
plt.ylabel('Cases Prediction')
plt.legend()
plt.show()
```


```python
new_df = data_clean[-sequence_length:]
N = sequence_length
```


```python
# Get the last N day closing price values and scale the data to be values between 0 and 1
last_N_days = new_df[-sequence_length:].values
last_N_days_scaled = scaler.transform(last_N_days)
```


```python
# Create an empty list and Append past N days
X_test_new = []
X_test_new.append(last_N_days_scaled)

# Convert the X_test data set to a numpy array and reshape the data
pred_cases_scaled = model_from_saved_checkpoint.predict(np.array(X_test_new))
pred_cases_unscaled = scaler_pred.inverse_transform(pred_cases_scaled.reshape(-1, 1))
```


```python
# Print last price and predicted price for the next day
cases_today = np.round(new_df['new_cases_smoothed'][-1])
predicted_cases = np.round(pred_cases_unscaled.ravel()[0])
change_percent = np.round(100 - (cases_today * 100)/predicted_cases)
```


```python
plus = '+'; minus = ''
print(f'The close covid cases count today is  {cases_today}')
print(f'The predicted case count for the next day is {predicted_cases} ({plus if change_percent > 0 else minus}{change_percent}%)')
```


```python
!jupyter nbconvert covid_analysis.ipynb --to markdown --NbConvertApp.output_files_dir=.
!cat covid_analysis.md | tee -a index.md
!rm covid_analysis.md
```
