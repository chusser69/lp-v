
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn import metrics


(x_train, y_train), (x_test, y_test) = mnist.load_data()


plt.imshow(x_train[0], cmap='gray')

plt.show()

print(x_train[0])

# Reshape the data to fit the model
print("X_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", x_test.shape)
print("y_test shape", y_test.shape)

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')

x_test = x_test.astype('float32')
x_train /= 255 # Each image has Intensity from 0 to 255
x_test /= 255

x_train[0]

# Convert class vectors to binary class matrices
num_classes = 10
y_train = np.eye(num_classes)[y_train] # Return a 2-D array with ones on the diagonal and zeros elsewhere
y_test = np.eye(num_classes)[y_test] # f your particular categories is present then it mark as 1 else 0 in remain row

# Define the model architecture
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,))) # Input cosist of 784 Neuron ie 784 input,512 in the hidden layer
model.add(Dropout(0.2)) # DROP OUT RATIO 20%

model.add(Dense(512, activation='relu')) #returns a sequence of another vectors of dimension 512
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax')) # 10 neurons ie output node in the output layer.
# Compile the model
model.compile(loss='categorical_crossentropy', # for a multi-class classification problem
optimizer=RMSprop(),
metrics=['accuracy'])

# Train the model
batch_size = 128 # batch_size argument is passed to the layer to define a batch size for the inputs.
epochs = 20
history = model.fit(x_train, y_train,
batch_size=batch_size,
epochs=epochs,
verbose=1, # verbose=1 will show you an animated progress bar eg. [==========]
validation_data=(x_test, y_test)) # Using validation_data means you are providing the training set and validation set yourself,
# 60000image/128=469 batch each

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
     




















# 📥 1. Importing Required Libraries

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn import metrics
```
- **NumPy** for numerical calculations.
- **Keras** from TensorFlow to build the neural network.
- **Matplotlib** for visualisation.
- **Scikit-learn (metrics)** was imported but is unused here.

---

# 📚 2. Loading the MNIST Dataset

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
- **MNIST** is a dataset of **28×28 pixel** grayscale images of handwritten digits (0–9).

Visualising the first image:

```python
plt.imshow(x_train[0], cmap='gray')
plt.show()
print(x_train[0])
```
- Displays the first image from the training set.
- Also prints the **raw pixel values**.

---

# 🧹 3. Checking the Shape of Data

```python
print("X_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", x_test.shape)
print("y_test shape", y_test.shape)
```
- Training Set: 60,000 images and labels.
- Testing Set: 10,000 images and labels.

Shape before preprocessing:  
- `x_train`: (60000, 28, 28)  
- `x_test`: (10000, 28, 28)

---

# 🧹 4. Preprocessing the Images

Flattening the 28×28 images into 784-dimensional vectors:

```python
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
```
Normalising pixel values from [0, 255] to [0, 1]:

```python
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
```
This helps the neural network to converge faster.

---

# 🧹 5. One-Hot Encoding of Labels

```python
num_classes = 10
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]
```
- Converts scalar labels like 3 → [0,0,0,1,0,0,0,0,0,0].
- Required for **categorical crossentropy** loss.

---

# 🏗️ 6. Defining the Neural Network Model

```python
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
```
- **Input Layer**: 784 neurons (pixels).
- **Two Hidden Layers**: Each with 512 neurons, using **ReLU** activation.
- **Dropout Layers**: 20% dropout to avoid overfitting.
- **Output Layer**: 10 neurons with **softmax** activation for multi-class probability outputs.

---

# ⚙️ 7. Compiling the Model

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy']
)
```
- **Loss Function**: Categorical Crossentropy (suitable for multi-class classification).
- **Optimizer**: RMSprop (adaptive learning rate).
- **Metric**: Accuracy.

---

# 🏋️ 8. Training the Model

```python
batch_size = 128
epochs = 20
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test)
)
```
- Trains the model for **20 epochs** with a batch size of **128**.
- Uses the **test data** for validation after each epoch.

---

# 📈 9. Evaluating the Model

```python
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```
- **Test Loss**: The error value on the test dataset.
- **Test Accuracy**: The classification accuracy on unseen data.

---

