import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

def gaussian_2d(x, y, sigma, amplitude=1):
    return amplitude * np.exp(-0.5 * ((x**2 / sigma**2) + (y**2 / sigma**2)))

# Constants
image_size = 400
deg_to_pixels = image_size / 15.04
sigma_values = [0.226 * deg_to_pixels, 0.301 * deg_to_pixels]
contrast_values = [-0.109, -0.141, -0.172, 0.109, 0.141, 0.172]
num_gaussian_blobs = 30
max_rms_contrast = 0.156

def generate_gaussian_blob_positions(num_blobs, image_size, min_sigma, max_sigma, min_contrast, max_contrast):
    blobs = []
    for _ in range(num_blobs):
        x0, y0 = np.random.randint(0, image_size, 2)
        sigma = np.random.uniform(min_sigma, max_sigma)
        contrast = np.random.uniform(min_contrast, max_contrast)
        blobs.append((x0, y0, sigma, contrast))
    return blobs

def generate_background(image_size, blobs):
    background = np.zeros((image_size, image_size))
    for x0, y0, sigma, contrast in blobs:
        x, y = np.meshgrid(np.arange(image_size) - x0, np.arange(image_size) - y0)
        blob = gaussian_2d(x, y, sigma, amplitude=contrast)
        background += blob
    return background

def shift_blobs(blobs, shift_x, shift_y, image_size):
    shifted_blobs = []
    for x0, y0, sigma, contrast in blobs:
        new_x = (x0 + shift_x) % image_size
        new_y = (y0 + shift_y) % image_size
        shifted_blobs.append((new_x, new_y, sigma, contrast))
    return shifted_blobs

def add_white_noise(image, rms_contrast):
    noise = np.random.normal(0, rms_contrast, image.shape)
    return image + noise

def place_gaussian_target(image, sigma, contrast):
    x0 = np.random.randint(0, image_size)
    y0 = np.random.randint(0, image_size)
    
    x, y = np.meshgrid(np.arange(image_size) - x0, np.arange(image_size) - y0)
    target = gaussian_2d(x, y, sigma, amplitude=contrast)

    image += target  # Directly add to the entire image
    return image, (x0 / image_size, y0 / image_size)

def make_images(num_images, rms_contrast, blobs):
    images, labels = [], []
    for _ in range(num_images):
        shift_x = np.random.randint(-image_size, image_size)
        shift_y = np.random.randint(-image_size, image_size)
        shifted_blobs = shift_blobs(blobs, shift_x, shift_y, image_size)
        background = generate_background(image_size, shifted_blobs)
        background_with_noise = add_white_noise(background, rms_contrast)
        target_sigma = np.random.choice(sigma_values)
        target_contrast = np.random.choice(contrast_values)
        image_with_target, target_position = place_gaussian_target(background_with_noise, target_sigma, target_contrast)
        images.append(np.clip(image_with_target + 0.5, 0, 1))
        labels.append(target_position)
    return np.array(images, dtype=np.float32), np.array(labels)

# Generate the dataset
print("generating dataset")
common_blobs = generate_gaussian_blob_positions(num_gaussian_blobs, image_size, 0.188 * deg_to_pixels, 0.564 * deg_to_pixels, -0.172, 0.172)
num_images = 1000
images, labels = make_images(num_images, 0, common_blobs)

# Normalize and reshape images
images = images.reshape((images.shape[0], image_size, image_size, 1))

# Split the dataset
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.1, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

# Build the CNN
print("Start model")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(400, 400, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='sigmoid')
])

# Compile the model
print("Compile model")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
print("train model")
history = model.fit(train_images, train_labels, epochs=20, batch_size=32, validation_data=(val_images, val_labels))

# Evaluate the model
print("evaluate model")
test_loss = model.evaluate(test_images, test_labels)
print(f'Test Loss: {test_loss}')

# Predict and visualize a limited number of images
num_to_display = 10  # Adjust this number as needed
predictions = model.predict(test_images)
for i in range(num_to_display):
    plt.imshow(test_images[i].reshape(400, 400), cmap='gray')
    plt.scatter([predictions[i][0] * 400], [predictions[i][1] * 400], color='red', marker='x', label='Predicted')
    plt.scatter([test_labels[i][0] * 400], [test_labels[i][1] * 400], color='blue', marker='o', label='Actual')
    plt.title(f"Predicted: {predictions[i]}, Actual: {test_labels[i]}")
    plt.legend()
    plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

#test changes 1,2
#test changes 3