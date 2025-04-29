# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Import the data
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Create data structure
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Exploratory Data Analysis
plt.figure(figsize=(18, 8))
plt.plot(dataset_train['Open'])
plt.title("Google Stock Open Prices")
plt.xlabel("Time (oldest -> latest)")
plt.ylabel("Stock Open Price")
plt.show()

plt.figure(figsize=(18, 8))
plt.plot(dataset_train['Low'])
plt.title("Google Stock Low Prices")
plt.xlabel("Time (oldest -> latest)")
plt.ylabel("Stock Lowest Price")
plt.show()

# Create model
regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fit model
regressor.fit(x=X_train, y=y_train, batch_size=32, epochs=100)

















Of course! I'll explain it step-by-step clearly, in **British English**:

---

### 1. **Importing Libraries**
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
```
- **NumPy**: for numerical operations (especially arrays).
- **Matplotlib**: for plotting graphs.
- **Pandas**: for handling datasets (reading CSV files, managing tables).
- **MinMaxScaler**: to scale data between 0 and 1, important for neural networks.
- **Keras (Tensorflow)**: to build the LSTM model.

---

### 2. **Importing the Dataset**
```python
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
```
- Loads the stock prices from a CSV file.
- `iloc[:, 1:2]` selects only the 'Open' column (the second column) as a NumPy array.

---

### 3. **Feature Scaling**
```python
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
```
- Scales all stock prices into the range **[0, 1]**.
- This improves the performance and speed of the LSTM.

---

### 4. **Creating the Data Structure**
```python
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
```
- **Input (X_train)**: last 60 stock prices.
- **Output (y_train)**: the next stock price after those 60 days.
- This creates a "look back" period of 60 time steps to predict the 61st.

---

### 5. **Reshaping the Input**
```python
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
```
- Reshapes the data into 3D format, required for LSTM:
  - **samples**, **time steps**, **features**

---

### 6. **Exploratory Data Analysis (EDA)**
```python
plt.figure(figsize=(18, 8))
plt.plot(dataset_train['Open'])
plt.title("Google Stock Open Prices")
plt.xlabel("Time (oldest -> latest)")
plt.ylabel("Stock Open Price")
plt.show()
```
- Plots the stock price to visually understand the trend.
- Similar code is used for the **Low** prices as well.

---

### 7. **Building the LSTM Model**
```python
regressor = Sequential()
```
- Initialises a **Sequential** model (layer-by-layer).

```python
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
```
- Adds an **LSTM layer** with 50 neurons, returning sequences for stacking.
- Adds a **Dropout layer** (20% of neurons dropped) to prevent overfitting.

This structure is **repeated three more times** (stacked LSTMs).

Finally:
```python
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))
```
- The last LSTM does not return sequences (as it’s the last one).
- **Dense layer** outputs one value: the predicted stock price.

---

### 8. **Compiling the Model**
```python
regressor.compile(optimizer='adam', loss='mean_squared_error')
```
- **Adam**: an efficient optimiser.
- **Mean Squared Error**: the loss function suitable for regression.

---

### 9. **Training the Model**
```python
regressor.fit(x=X_train, y=y_train, batch_size=32, epochs=100)
```
- Trains the model with:
  - **32 samples** per batch,
  - **100 full passes** (epochs) over the dataset.

---

### Simple Flowchart:
1. Load and scale stock prices.
2. Create sequences of 60 prices to predict the next price.
3. Build a stacked LSTM model with dropout.
4. Train the model on the data.
5. Model becomes capable of predicting future stock prices!

---
