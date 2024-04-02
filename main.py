from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


# Function to extract dominant colors from an image
def extract_colors(image_path, num_colors):
    # Load image
    image = Image.open(image_path)
    # Convert image to RGB
    image = image.convert('RGB')
    # Convert image to numpy array
    image_array = np.array(image)
    # Flatten the array to 2D (pixels x RGB)
    flattened_array = image_array.reshape((-1, 3))
    # Use K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(flattened_array)
    # Get the RGB values of the cluster centers (dominant colors)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    formatted_colors = [', '.join(map(str, color)) for color in dominant_colors]
    return formatted_colors


# Paths to your PNG images
front_image_path = 'merc-front.png'
right_image_path = 'merc-right-side.png'
left_image_path = 'merc-left.png'
back_image_path = 'merc-back.png'

# Number of dominant colors to extract
num_colors = 5

# Extract dominant colors from each image
front_colors = extract_colors(front_image_path, num_colors)
right_colors = extract_colors(right_image_path, num_colors)
left_colors = extract_colors(left_image_path, num_colors)
back_colors = extract_colors(back_image_path, num_colors)

# Optionally, visualize the extracted colors
# You can use matplotlib or any other plotting library for this

# Display or save the results as needed
print(f"Front colors: {front_colors}")
print(f"Right colors: {right_colors}")
print(f"Left colors: {left_colors}")
print(f"Back colors: {back_colors}")
