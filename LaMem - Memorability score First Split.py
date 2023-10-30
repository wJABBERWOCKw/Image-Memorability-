#!/usr/bin/env python
# coding: utf-8

# *Loading Necessary Libraries*

# In[ ]:


import tensorflow as tf
from tensorflow.keras.applications import ResNet18
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
import random
import multiprocessing as mp
import csv


# *Checking for the available GPU Devices*

# In[ ]:


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    print("GPU is available.")
else:
    print("No GPU available, using CPU.")


# *Data Processing*

# In[ ]:


import os
from sklearn.model_selection import train_test_split

# Function to load data split from lamem
def load_split(split_file):
    with open(split_file, 'r') as file:
        contents = [line.split() for line in file]
        contents = [[x[0], float(x[1])] for x in contents]
    return contents

# Function to load a single image from the 'images/' folder based on image number
def load_image_by_number(image_number):
    image_path = os.path.join('images/', f'{image_number}.jpg')
    return np.array(Image.open(image_path).resize((224, 224)).convert("RGB"), dtype="float32") / 255.

# Function to generate data for training and testing
def lamem_generator(train_split, test_split, batch_size):
    X_train = []
    Y_train = []
    for image_filename, memorability_score in train_split:
        image_number = image_filename.split('.')[0]
        image = load_image_by_number(image_number)
        X_train.append(image)
        Y_train.append(memorability_score)
    
    X_test = []
    Y_test = []
    for image_filename, memorability_score in test_split:
        image_number = image_filename.split('.')[0]
        image = load_image_by_number(image_number)
        X_test.append(image)
        Y_test.append(memorability_score)

    return (np.array(X_train), np.array(Y_train)), (np.array(X_test), np.array(Y_test))





# *Loading the first split*

# In[ ]:


# Load the first train/test split
train_split = load_split("splits/train_1.txt")
test_split = load_split("splits/test_1.txt")

# Generate data for training and testing
(X_train, Y_train), (X_test, Y_test) = lamem_generator(train_split, test_split, batch_size=batch_size)

# Split the data into train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


# *Resnet-18*

# In[ ]:


import tensorflow as tf
from tensorflow.keras.applications import ResNet18
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# Load the pre-trained ResNet-18 model without the top layer
base_model = ResNet18(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom top layers for memorability prediction
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1)(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

if len(physical_devices) > 0:
    model = multi_gpu_model(model, gpus=len(physical_devices))


# *Training for the First split*

# In[ ]:


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=5, batch_size=batch_size)
test_loss = model.evaluate(X_test, Y_test, batch_size=batch_size)
print(f'Test Loss: {test_loss}')


# In[ ]:




