"""
Facealytics by Rishit Pant.
"""
# Data Variables | Dataset from Kaggle
train_data = "./face_data/DATA/train/"
test_data = "./face_data/DATA/testing/"
list_train = ['Acne', 'Rosacea', 'Eczemaa']  # The category we want to train the Model with.

# Import Libraries | pip install requirements.txt
from PIL import Image as PILImageHandler
from tkinter import *; from tkinter import ttk
import ctypes, subprocess
import os, cv2, numpy as np, pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.imagenet_utils import preprocess_input

VGG = VGG19(include_top=False, weights='imagenet')  # Load pre-trained VGG19 model trained on ImageNet database
VGG.summary()

def data_gen(): # This module will scour through defined paths to find images & add them to our dictionary.
    k = 0
    # Dictionaries to store training & testing image paths and targets
    train_dictionary = {"img_path": [], "target": []}
    test_dictionary = {"img_path": [], "target": []}  

    for i in list_train:
        path_disease_train = train_data + i
        path_disease_test = test_data + i

        image_list_train = os.listdir(path_disease_train)
        image_list_test = os.listdir(path_disease_test)
        """
        Working of the next 2 loops:
            1: We get the complete path to a training/testing image
            2: Add image path to the training/testing dictionary
            3: Add the target/testing category index to the training dictionary
        """
        for j in image_list_train:
            img_path_train = path_disease_train + "/" + j
            train_dictionary["img_path"].append(img_path_train)
            train_dictionary['target'].append(k)

        for m in image_list_test:
            img_path_test = path_disease_test + "/" + m
            test_dictionary["img_path"].append(img_path_test)
            test_dictionary['target'].append(k)
        k += 1

    # Create a testing & training DataFrame from the test & train dictionary.
    test_df = pd.DataFrame(test_dictionary)
    train_df = pd.DataFrame(train_dictionary)
    return train_df, test_df

def load_data(input_size=(100, 100)):  # Function to load and preprocess the data
    images = []
    images2 = []
    train_df, test_df = data_gen()
    """
    Working of the next 2 loops:
        1: Resize the image to the specified input size
        2: Append the resized image to the list of training/testing images
        3: Convert the target values to a NumPy array
        4: Convert the testing images list to a NumPy array
    """
    for i in train_df['img_path']:
        img = cv2.imread(i)
        img = cv2.resize(img, input_size)
        images.append(img)
    y_train = np.asarray(train_df['target'])
    x_train = np.asarray(images)

    for i in test_df['img_path']:
        img = cv2.imread(i)
        img = cv2.resize(img, input_size)
        images2.append(img)
    y_test = np.asarray(test_df['target'])
    x_test = np.asarray(images2)

    return x_train, x_test, y_train, y_test  # Return the preprocessed data

x_train, x_test, y_train, y_test = load_data(input_size=(100, 100))  # Load and preprocess the data

def load_img(img_path):  
    images = []
    img = cv2.resize(cv2.imread(img_path), (100, 100))
    images.append(img)
    x_test = np.asarray(images)

    test_img = preprocess_input(x_test)
    features_test = VGG.predict(test_img)
    num_test = x_test.shape[0]
    f_img = features_test.reshape(num_test, 4608)

    return f_img

# filename = "dbfile.sav"
# joblib.dump(VGG, open(filename, 'wb'))
# VGGLoad = joblib.load("dbfile.sav")

# Preprocess the Training & Testing images.
train_img = preprocess_input(x_train)
test_img = preprocess_input(x_test)

# Extract features from the preprocessed training & testing images
features_train = VGG.predict(train_img)  
features_test = VGG.predict(test_img)

# Get the number of training & testing images.
num_train = x_train.shape[0]
num_test = x_test.shape[0] 

# Reshape the Training & testing features.
x_train = features_train.reshape(num_train, 4608)
x_test = features_test.reshape(num_test, 4608)

# While Initializing the layer, we are specifying how many neurons it will use.
model = Sequential([
    Dense(1024, activation='relu'),  # Fully connected layer with 1024 units and ReLU activation...
    Dense(512, activation='relu'), Dense(256, activation='relu'), Dense(128, activation='relu'),
    Dense(64, activation='sigmoid')])  # Fully connected layer with 64 units and sigmoid activation

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=40, validation_data=(x_test, y_test))

plt.plot(model.history.history['accuracy'], label="train_accuracy")
plt.plot(model.history.history['val_accuracy'], label="validation_accuracy")
plt.legend()
plt.show()

# model.save('saved_model/skin_model')
# model = load_model('saved_model/skin_model')
img = load_img("face_data/DATA/train/Rosacea/rosacea-70.jpg")
np.argmax(model.predict(img))  # Make a prediction using the trained model.

# Classify a new image
img_path = "face_data/DATA/train/Rosacea/rosacea-70.jpg"

img = np.expand_dims(cv2.resize(cv2.imread(img_path), (100, 100)), axis=0)
img_features = VGG.predict(preprocess_input(img)).reshape(1, 4608)
prediction = np.argmax(model.predict(img_features))

# Convert the predicted class back to its original label
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
predicted_label = label_encoder.inverse_transform([prediction])[0]
encoded_y_train = label_encoder.transform(y_train)
encoded_y_test = label_encoder.transform(y_test)

#predicted_label = np.argmax(prediction)
predicted_class = list_train[predicted_label]

# Print the predicted class

def openImage():
    im = PILImageHandler.open(f"{img_path}")
    im.show()

root = Tk()
frm = ttk.Frame(root, padding=20)
frm.grid()
ttk.Label(frm, text=f"The predicted class for {img_path} is {list_train[0]}").grid(column=0, row=0)
ttk.Button(frm, text="See image", command= openImage).grid(column=1, row=0)
root.mainloop()

print(f"The predicted class for {img_path} is {list_train[predicted_label]}")