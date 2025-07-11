import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#* Define your dataset path
DATA_DIR = 'data/raw/asl_alphabet_train'

#* Define Image Size and Batch Size:
IMG_SIZE = (64,64)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.2,  # 80% train, 20% validation
    rotation_range = 10,
    zoom_range = 0.1,
    shear_range = 0.1,
    horizontal_flip = True
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    subset = 'training',
    shuffle = True 
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode ='categorical',
    subset = 'validation',
    shuffle = True
)

# Visualize a few Steps
x_batch, y_batch = next(train_generator)

plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3, i+1)
    plt.imshow(x_batch[i]) 
    plt.title(f"Label: {np.argmax(y_batch[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()