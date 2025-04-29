




import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
# Normalize
train_images = train_images / 255.0
test_images = test_images / 255.0



plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([]); plt.yticks([]); plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()



model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10) # Output logits
])


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


model.fit(train_images, train_labels, epochs=30)



test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)




probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

predictions = probability_model.predict(test_images)





def plot_image(i, predictions_array, true_label, img):
    ...
    
def plot_value_array(i, predictions_array, true_label):
    ...




































# 🌟 Project: **Fashion MNIST Clothing Classification using CNN**

## 1. Problem Statement
Develop a Convolutional Neural Network (CNN) based system to classify images of clothing items from the Fashion MNIST dataset into 10 predefined categories.

## 2. Introduction
Fashion MNIST is a dataset of Zalando's article images consisting of 60,000 training examples and 10,000 test examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.  
Compared to original MNIST, which is a dataset of handwritten digits, Fashion MNIST poses a slightly more complex challenge and serves as a better benchmark for evaluating machine learning models.

## 3. Tools and Libraries
- TensorFlow 2.11.0
- NumPy
- Matplotlib

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
# Output: 2.11.0
```

---

## 4. Dataset Description
The dataset includes:
- 60,000 training images and 10,000 testing images.
- Each image is 28x28 pixels, grayscale.
- Labels correspond to clothing categories:
  ```python
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  ```

---

## 5. Data Preprocessing
- **Normalization**: Scale pixel values between 0 and 1 by dividing by 255.
- **Visualization**: Display sample images to ensure correctness.

```python
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize
train_images = train_images / 255.0
test_images = test_images / 255.0
```

Visualization example:

```python
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([]); plt.yticks([]); plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```

---

## 6. Model Architecture
Built a **simple CNN-based classifier** using the following layers:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10) # Output logits
])
```

- **Flatten** layer: Converts 28x28 images into a 1D array of 784 pixels.
- **Dense** (fully connected) layers:
  - 128 neurons with ReLU activation.
  - Output layer with 10 neurons (no activation initially).

---

## 7. Model Compilation
```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```
- Optimizer: **Adam**
- Loss Function: **Sparse Categorical Crossentropy** (suitable for integer labels)
- Evaluation Metric: **Accuracy**

---

## 8. Model Training
Training for 30 epochs on the training set.

```python
model.fit(train_images, train_labels, epochs=30)
```

- Achieved final **training accuracy: ~94.6%** after 30 epochs.

---

## 9. Model Evaluation
Testing the model on unseen data:

```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
```

- **Test Accuracy**: ~87.9%
- **Observation**: Slight overfitting observed (training accuracy > testing accuracy).

---

## 10. Making Predictions
After training:

```python
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

predictions = probability_model.predict(test_images)
```
Each prediction returns a probability distribution across 10 clothing categories.

---

## 11. Visualization of Predictions
Functions to visualize prediction results:

```python
def plot_image(i, predictions_array, true_label, img):
    ...
    
def plot_value_array(i, predictions_array, true_label):
    ...
```

Example:

```python
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()
```

---

## 12. Conclusion
- Successfully implemented a CNN-based classifier for Fashion MNIST dataset.
- Achieved **~88% accuracy** on the test set.
- Model slightly overfits; further improvements like dropout, data augmentation, or advanced architectures (e.g., CNN layers) could help improve performance.

---
