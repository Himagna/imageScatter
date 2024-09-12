# Number of clusters
num_clusters = 10


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

image_path = 'input.jpg'
image = Image.open(image_path)

image_array = np.array(image)

# Reshape the image array to a 2D array of RGB values
pixels = image_array.reshape(-1, 3)

# k-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(pixels)
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

# plotting
x_coords = []
y_coords = []
colors = []
sizes = []

for cluster_index in range(num_clusters):
    # indices of pixels in the current cluster
    cluster_indices = np.where(labels == cluster_index)[0]
    
    # average x and y coordinates for this cluster
    cluster_y, cluster_x = np.divmod(cluster_indices, image_array.shape[1])
    avg_x = np.mean(cluster_x)
    avg_y = np.mean(cluster_y)
    
    # average color for this cluster
    avg_color = cluster_centers[cluster_index] / 255
    
    # size of the scatter point based on the number of pixels in the cluster
    point_size = len(cluster_indices)
    
    # Append the data for plotting
    x_coords.append(avg_x)
    y_coords.append(-avg_y)  # Invert y to match image orientation
    colors.append(avg_color)
    sizes.append(point_size)

# Plotting
plt.figure(figsize=(10, 10))
plt.scatter(x_coords, y_coords, c=colors, s=sizes, alpha=0.6)
plt.axis('off')
plt.savefig('output.jpg')
