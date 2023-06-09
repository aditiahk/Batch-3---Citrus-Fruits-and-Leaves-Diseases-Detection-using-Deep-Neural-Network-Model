# set the matplotlib backend so figures can be saved in the background

# import the necessary packages
import os
import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from imutils import paths
import argparse
import pickle
import h5py
LABELS = set(["Black spot", "canker", "greening","healthy","melanose","Scab"])
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
dataset='./dataset/'
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset))
#print(imagePaths)
data = []
labels = []
# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]
	label = label.split('/')[-1]
	# if the label of the current image is not part of of the labels
	# are interested in, then ignore the image
	if label not in LABELS:
		continue
	# load the image, convert it to RGB channel ordering, and resize
	# it to be a fixed 224x224 pixels, ignoring aspect ratio
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
	#cv2.imshow("image",image)
	# update the data and labels lists, respectively
	data.append(image)
	labels.append(str(label))

# convert the data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.2, stratify=labels, random_state=42)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(224, 224, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(224, 224, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
)
epoch=10
history = model.fit(
        x=train_datagen.flow(trainX, trainY, batch_size=64),
        steps_per_epoch=len(trainX) // 64,
        validation_data=valid_datagen.flow(testX, testY),
        epochs=epoch)


    
print("[INFO] evaluating network...")
predictions = model.predict(x=testX.astype("float64"), batch_size=64)

# confusion matrix
print("Confusion matrix")
print(confusion_matrix(testY.argmax(axis=1),
	predictions.argmax(axis=1)))


cm= confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))
ax = sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
## Display the visualization of the Confusion Matrix.
plt.savefig("Confusion_matrix.png")
# plot the training loss and accuracy
plt.show()
N = epoch
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['accuracy'])
#plt.plot(history_1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("taining_acc.png")
plt.show()

# summarize history for loss

plt.plot(history.history['loss'])
#plt.plot(history_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("taining_loss.png")
plt.show()



# serialize the model to disk
print("[INFO] serializing network...")
model.save("test.h5")
# serialize the label binarizer to disk
pkl_filename = "test.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(lb, file)
