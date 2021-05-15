import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2  
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import seaborn as sns
import pathlib
import PIL
import numpy as np
import dlib

batchSize = 30
imgHeight = 180
imgWidth = 180

# Initialising Face Detector
detector = dlib.get_frontal_face_detector()
imgD = cv2.imread('test4.jpeg')
imgD = cv2.resize(imgD, (0, 0), fx=0.25, fy=0.25)
gray = cv2.cvtColor(imgD, cv2.COLOR_BGR2GRAY)
rect = detector(gray, 0)


print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Loading Image Dataset

Class_names=['Dusky','Fair']
dataset = pathlib.Path('StrDataSet')
trainDS=tf.keras.preprocessing.image_dataset_from_directory(dataset,labels='inferred',class_names=Class_names,label_mode="int",seed=60,validation_split=0.2,subset='training',image_size=(imgHeight, imgWidth),batch_size=batchSize)
validationDS=tf.keras.preprocessing.image_dataset_from_directory(dataset,labels='inferred',class_names=Class_names,label_mode="int",seed=60,validation_split=0.2,subset='validation',image_size=(imgHeight, imgWidth),batch_size=batchSize)
"""Fair = list(dataset.glob('Dusky/*.jpg'))
im=PIL.Image.open(str(Fair[0]))
im.show()

for image,label in trainDS:
    print(image.shape)
    print(label.shape)
    break
"""

# Building a model

num_classes = 2

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(imgHeight, imgWidth, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=10
history = model.fit(trainDS,validation_data=validationDS,epochs = epochs)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

"""
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
"""
# Testing Image

Test = pathlib.Path('test4.jpeg')
testData=tf.keras.preprocessing.image.load_img(Test,target_size=(imgHeight,imgWidth))
imgArray=tf.keras.preprocessing.image.img_to_array(testData)
imgArray=tf.expand_dims(imgArray,0)

predict=model.predict(imgArray)
score = tf.nn.softmax(predict[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(Class_names[np.argmax(score)], 100 * np.max(score))
)

for i,d in enumerate(rect):
    x1, y1, x2, y2, h, w = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.height(), d.width()

msg=Class_names[np.argmax(score)]+" with a {:.2f} sureity ".format(100 * np.max(score))
cv2.rectangle(imgD, (x1, y1), (x2, y2), (255, 150, 67), 2)
cv2.putText(imgD, msg, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 245), 1)
cv2.imshow("Results", imgD)
cv2.waitKey(0)
cv2.destroyAllWindows()
