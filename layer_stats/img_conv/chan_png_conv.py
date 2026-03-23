import numpy as np
from scipy.io import savemat
import cv2
import os

# Function to read and process black and white images in a directory
def process_images(directory, target_size):
    # Initialize an empty list to store processed images
    processed_images = []

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):  # Check if the file is a JPEG image
            # Read the image
            image = cv2.imread(os.path.join(directory, filename))

            # Resize the image to the target size
            resized_image = cv2.resize(image, target_size)

            # Convert the resized image to grayscale
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

            # Threshold the grayscale image to create a binary image
            _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

            # Normalize the binary image to set white regions to 0 and other regions to 1
            binary_image = (binary_image > 128).astype(np.uint8)

            # Append the processed binary image to the list
            processed_images.append(binary_image)
    return processed_images

# Directory containing the images
directory = 'WWU_Data/WWU_Data/Chan_Maps/'

# Target size for resizing images
target_size = (522,796)  # Set the desired target size

# Process images in the directory
processed_images = process_images(directory, target_size)


# Stack processed images along the z-axis
stacked_images = np.stack(processed_images, axis=2)

# Save the stacked images to a .mat file
savemat('channel.mat', {'channel': stacked_images})

np.savetxt('test.txt', stacked_images[:,:,0])
print(stacked_images.shape)
