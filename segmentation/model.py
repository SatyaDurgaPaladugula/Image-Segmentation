from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
image = plt.imread('balls.jpeg')
print(image.shape)

# Convert the image to grayscale
gray = rgb2gray(image)

# Scale the grayscale image to the range 0-255 (uint8)
gray_scaled = (gray * 255).astype(np.uint8)

# Display the grayscale image
plt.imshow(gray_scaled, cmap='gray')
plt.axis('off')  # Turn off axis labels
plt.show()

# Threshold segmentation with multiple thresholds
gray_r = gray.reshape(gray.shape[0]*gray.shape[1])

# Define threshold values
thresholds = [0, 0.25, 0.5, gray.mean()]

# Create instance image based on thresholds
instance_image = np.zeros_like(gray_r, dtype=np.uint8)

for i in range(len(thresholds)):
    instance_image[gray_r > thresholds[i]] = i + 1

# Reshape instance image to original shape
instance_image = instance_image.reshape(gray.shape[0], gray.shape[1])

# Display the instance image with colormap
plt.imshow(instance_image, cmap='jet')
plt.colorbar()  # Add colorbar to show threshold levels
plt.axis('off')  # Turn off axis labels
plt.show()