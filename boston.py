

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
BostonTrain = pd.read_csv('boston_train.csv')


BostonTrain.head()

BostonTrain.info()
BostonTrain.describe()

#ID columns is not relevant for our analysis.
BostonTrain.drop('ID', axis = 1, inplace=True)

BostonTrain.plot.scatter('rm', 'medv')

# Now lets take a look how all variables relate to each other.
plt.subplots(figsize=(12,8))
sns.heatmap(BostonTrain.corr(), cmap = 'RdGy')

# Lets plot the paiplot, for all different correlations
# Negative Correlation: When x is high y is low and vice versa., To the right less negative correlation.
sns.pairplot(BostonTrain, vars = ['lstat', 'ptratio', 'indus', 'tax', 'crim', 'nox', 'rad', 'age', 'medv'])


X = BostonTrain[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'black', 'lstat']]
y = BostonTrain['medv']


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


sns.displot((y_test-predictions),bins=50)



# ======== Neural Networks  =============================================

#Scaling the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Creating the neural network model
# pip install --upgrade pip
# pip install keras
# pip install tensorflow
# pip show tensorflow

from tensorflow.keras.models import Sequential
model = Sequential()

from tensorflow.keras.layers import Dense
model.add(Dense(units=128, activation='relu', input_shape=(13,)))
model.add(Dense(units = 64, activation='relu'))
model.add(Dense(units = 32, activation='relu'))
model.add(Dense(units = 16, activation='relu'))
model.add(Dense(1))
model.summary()
model.compile(optimizer = 'adam', loss='mean_squared_error', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=15, verbose=1, validation_split=0.05)
#Evaluation of the model
y_pred = model.predict(X_test)
mse_nn, mae_nn = model.evaluate(X_test, y_test)
print('Mean absolute error on test data using NN: ', mae_nn)
print('Mean squared error on test data using NN: ', mse_nn)
print('RMSE using NN:', np.sqrt(mse_nn))

plt.scatter(y_test,y_pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
     

























# 🌟 Code Explanation (Boston Housing Dataset)

You are performing **regression analysis** on the **Boston Housing dataset**, using two approaches:
- Linear Regression (simple machine learning)
- Neural Network (deep learning)

---

# 📥 1. Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
- `pandas` for data handling
- `numpy` for numerical operations
- `matplotlib` and `seaborn` for plotting graphs and charts

---

# 📑 2. Loading and Inspecting the Data

```python
BostonTrain = pd.read_csv('boston_train.csv')
BostonTrain.head()
BostonTrain.info()
BostonTrain.describe()
```
- **Load** the training dataset.
- `head()` → shows first 5 rows.
- `info()` → shows data types and missing values.
- `describe()` → shows statistical summary.

---

# 🧹 3. Cleaning Data

```python
BostonTrain.drop('ID', axis=1, inplace=True)
```
- The `ID` column is **not useful** for modelling, so it is removed.

---

# 📊 4. Data Visualisation

- **Scatter Plot** (to see relationship between number of rooms and house price):

```python
BostonTrain.plot.scatter('rm', 'medv')
```

- **Heatmap of Correlations** (to check how strongly features are related):

```python
plt.subplots(figsize=(12,8))
sns.heatmap(BostonTrain.corr(), cmap='RdGy')
```

- **Pair Plot** (only for selected important features):

```python
sns.pairplot(BostonTrain, vars=['lstat', 'ptratio', 'indus', 'tax', 'crim', 'nox', 'rad', 'age', 'medv'])
```

---

# 📦 5. Preparing the Data

```python
X = BostonTrain[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']]
y = BostonTrain['medv']
```
- `X` → **Features** (inputs)
- `y` → **Target** (median value of home - `medv`)

---

# 📚 6. Linear Regression Model

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

- **Split** the dataset into training and testing parts:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
```
(40% test data)

- **Train** the model:

```python
lm = LinearRegression()
lm.fit(X_train, y_train)
```

- **Predict** on test data:

```python
predictions = lm.predict(X_test)
```

- **Plot** actual vs predicted values:

```python
plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
```

---

# 🧮 7. Model Evaluation (Linear Regression)

```python
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```
- MAE = Mean Absolute Error
- MSE = Mean Squared Error
- RMSE = Root Mean Squared Error

- **Distribution of residuals (errors):**

```python
sns.displot((y_test - predictions), bins=50)
```

---

# 🤖 8. Neural Network Model

- **Feature Scaling** is applied (important for neural networks):

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

- **Building Neural Network**:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

```python
model = Sequential()
model.add(Dense(units=128, activation='relu', input_shape=(13,)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(1))
```
- 4 hidden layers with ReLU activation.
- 1 output layer (for predicting house price).

- **Model Summary**:

```python
model.summary()
```

- **Compile the model**:

```python
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
```
- Optimiser: **Adam**
- Loss function: **Mean Squared Error (MSE)**

- **Train the model**:

```python
history = model.fit(X_train, y_train, epochs=15, verbose=1, validation_split=0.05)
```
- 15 epochs
- 5% data used for validation

---

# 🧪 9. Neural Network Evaluation

- **Predicting**:

```python
y_pred = model.predict(X_test)
```

- **Evaluating**:

```python
mse_nn, mae_nn = model.evaluate(X_test, y_test)
print('Mean absolute error on test data using NN: ', mae_nn)
print('Mean squared error on test data using NN: ', mse_nn)
print('RMSE using NN:', np.sqrt(mse_nn))
```

- **Plot actual vs predicted**:

```python
plt.scatter(y_test, y_pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
```

---

