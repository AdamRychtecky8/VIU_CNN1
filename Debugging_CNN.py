import numpy as np
import matplotlib.pyplot as plt

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

def shift_image(image, shift_x, shift_y):
    shifted_image = np.roll(image, shift_x, axis=1)
    shifted_image = np.roll(shifted_image, shift_y, axis=0)
    return shifted_image

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
    base_background = generate_background(image_size, blobs)
    images, labels = [], []  # Initialize as empty lists
    for _ in range(num_images):
        shift_x = np.random.randint(-image_size, image_size)
        shift_y = np.random.randint(-image_size, image_size)
        shifted_background = shift_image(base_background, shift_x, shift_y)
        background_with_noise = add_white_noise(shifted_background, rms_contrast)
        target_sigma = np.random.choice(sigma_values)
        target_contrast = np.random.choice(contrast_values)
        image_with_target, target_position = place_gaussian_target(background_with_noise, target_sigma, target_contrast)
        images.append(np.clip(image_with_target + 0.5, 0, 1))
        labels.append(target_position)
    return np.array(images, dtype=np.float32), np.array(labels)

# Generate the dataset
print("Generating dataset")
common_blobs = generate_gaussian_blob_positions(num_gaussian_blobs, image_size, 0.188 * deg_to_pixels, 0.564 * deg_to_pixels, -0.172, 0.172)
num_images = 5  # Generating only 5 images for visualization
images, labels = make_images(num_images, 0, common_blobs)

# Display the images and target locations
for i in range(num_images):
    plt.imshow(images[i], cmap='gray')
    plt.scatter([labels[i][0] * image_size], [labels[i][1] * image_size], color='red', marker='x', label='Target')
    plt.title(f"Image {i + 1} with Target Location")
    plt.legend()
    plt.show()
