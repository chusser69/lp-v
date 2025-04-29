import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten

# Set parameters
max_words = 10000 # Consider only the top 10,000 most frequently occurring words
max_length = 250 # Limit the review length to 250 words
embedding_size = 50 # Dimensionality of the word embeddings
# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

# Preprocess the data
#following function transforms a list of sequences into a 2D Numpy array
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)


# Build the model
model = Sequential()
#The Embedding layer is defined as the first hidden layer of a network.
model.add(Embedding(max_words, embedding_size, input_length=max_length))
#If you wish to connect a Dense layer directly to an Embedding layer, you must first flatten the 2D output matrix to a 1D vector using the Flatten layer.
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
batch_size = 32
epochs = 1
#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.05)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
















Here’s your **IMDB Sentiment Analysis** code explained properly in **British English**:

---

# 🖋 Full Code and Explanation (British English)

---

# 📥 1. Import Required Libraries

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten
```
- **NumPy** for numerical computations.
- **Keras** (from TensorFlow) to build and train the deep learning model.
- **IMDB dataset** provided directly by Keras.

---

# 📚 2. Set Parameters

```python
max_words = 10000  # Only the top 10,000 most common words are considered
max_length = 250   # Each review will be truncated or padded to 250 words
embedding_size = 50 # Each word is represented by a 50-dimensional vector
```
- We **limit the vocabulary size** and **fix the input length** to standardise data for the neural network.

---

# 📚 3. Load and Preprocess the Dataset

```python
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
```
- `x_train` and `x_test` contain lists of integers (word indices).
- `y_train` and `y_test` contain **binary labels** (0 = negative, 1 = positive).

**Pad the sequences** so that all reviews are exactly 250 words long:

```python
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)
```
- Short reviews are **padded with zeros**.
- Long reviews are **truncated** to 250 words.

---

# 🏗️ 4. Build the Neural Network Model

```python
model = Sequential()
model.add(Embedding(max_words, embedding_size, input_length=max_length))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
```
Explanation of layers:
- **Embedding Layer**: Converts word indices into dense vector representations (word embeddings).
- **Flatten Layer**: Flattens the output of the Embedding layer (2D to 1D).
- **Dense Layer (128 units, ReLU activation)**: Fully connected hidden layer.
- **Dropout Layer (50%)**: Prevents overfitting by randomly deactivating 50% of the neurons.
- **Dense Layer (1 unit, Sigmoid activation)**: Output layer for binary classification (positive/negative).

---

# ⚙️ 5. Compile the Model

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```
- **Loss Function**: Binary Crossentropy (suitable for two-class problems).
- **Optimizer**: Adam (adaptive learning rate optimisation).
- **Metrics**: Accuracy.

---

# 🏋️ 6. Train the Model

```python
batch_size = 32
epochs = 1
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
```
- **Batch Size**: 32 reviews at a time.
- **Epochs**: 1 pass through the full training data (can be increased for better performance).
- **Validation Data**: Uses the test set during training to monitor performance.

---

# 📈 7. Evaluate the Model

```python
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
```
