import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from keras_preprocessing import image

# ------------------------------------------------------------------------------------------------------------------
# Data Preprocessing
# ------------------------------------------------------------------------------------------------------------------
# Image Augmentation
training_datagen = ImageDataGenerator(
    rescale = 1./255, # Applies feature scaling to every pixel value (every pixel = 0-255 -> 0-1)
    shear_range = 0.2, # Applies shear/tilt to the image
    zoom_range = 0.2, # Random zoom in/out
    horizontal_flip = True # Random flips
)

# Import training set
training_set = training_datagen.flow_from_directory(
    './training_set',
    target_size = (64, 64), # Final size of the images when they will be fitted into the CNN
    batch_size = 32, # How many images do we want in each batch, 32 is a classic default value
    class_mode = 'binary' # Binary or categorical, in this case, we want to choose between cat or dog so binary
)

# Import test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    './test_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary'
)

# ------------------------------------------------------------------------------------------------------------------
# Build CNN
# ------------------------------------------------------------------------------------------------------------------
cnn = tf.keras.models.Sequential()

# Convolution
cnn.add(tf.keras.layers.Conv2D(
    filters = 32,
    kernel_size = (3, 3),
    activation = 'relu',
    input_shape = [64, 64, 3]
))

# Pooling (max)
cnn.add(tf.keras.layers.MaxPool2D(
    pool_size = 2,
    strides = 2
))

# 2nd convolution layer
cnn.add(tf.keras.layers.Conv2D(
    filters = 32,
    kernel_size = (3, 3),
    activation = 'relu'
))
cnn.add(tf.keras.layers.MaxPool2D(
    pool_size = 2,
    strides = 2
))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Full connection
cnn.add(tf.keras.layers.Dense(
    units = 128,
    activation = 'relu',
))

# Output layer
cnn.add(tf.keras.layers.Dense(
    units = 1,
    activation = 'sigmoid' # Because binary classification, for multi-class classification softmax
))

# ------------------------------------------------------------------------------------------------------------------
# Train CNN
# ------------------------------------------------------------------------------------------------------------------
# Compiling
cnn.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

# Training
cnn.fit(
    x = training_set,
    validation_data = test_set,
    epochs = 25
)

# ------------------------------------------------------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------------------------------------------------------
test_image = image.load_img(
    './single_prediction/cat.png',
    target_size = (64, 64)
)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
# print(training_set.class_indices)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)