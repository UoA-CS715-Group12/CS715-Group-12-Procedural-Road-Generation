import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN


def load_image(image_path):
    """Load an image from the given path and convert it to a binary black and white map."""
    # Load the image using Pillow
    image = Image.open(image_path)

    # Convert the image to grayscale
    image = image.convert('L')

    # Convert the image into a NumPy array
    image = np.array(image)

    # Threshold the image to create a binary black and white map
    # You can adjust the threshold value (128) to control the sensitivity
    image = (image > 70).astype(np.uint8)
    
    plt.show()
    
    return image

def extract_data_points(image):
    """Extract data points representing high-density areas from a binary black and white map."""
    # Create a list to store the data points
    data = []

    # Iterate over the pixels in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # If the pixel is white (i.e., high density)
            if image[i, j] == 1:
                # Add the pixel coordinates as a data point
                data.append([i, j])

    # Convert the data points into a NumPy array
    data = np.array(data)
    
    return data

def cluster_data_points(data, eps=6, min_samples=50):
    """Cluster data points using the DBSCAN algorithm."""
    # Create a DBSCAN clustering model
    db = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit the model to the data
    db.fit(data)

    # Get the cluster labels for each data point
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print(f'Estimated number of clusters: {n_clusters_}')
    
    return labels

def get_cluster_coordinates(data, labels):
    """Get the coordinates of each cluster."""
    # Create a list to store the cluster coordinates
    clusters = []

    # Iterate over the unique cluster labels
    for label in np.unique(labels):
        # If the label is not noise
        if label != -1:
            # Get the data points belonging to this cluster
            cluster_data = data[labels == label]

            # Add the cluster coordinates to the list
            clusters.append(cluster_data)
    
    return clusters

def get_cluster_centroids(data, labels):
    """Get the centroids of each cluster."""
    # Create a list to store the cluster centroids
    centroids = []

    # Iterate over the unique cluster labels
    for label in np.unique(labels):
        # If the label is not noise
        if label != -1:
            # Get the data points belonging to this cluster
            cluster_data = data[labels == label]

            # Compute the centroid of the cluster
            centroid = np.mean(cluster_data, axis=0)

            # Add the centroid to the list
            centroids.append(centroid)

    # Convert the centroids into a NumPy array
    centroids = np.array(centroids)
    
    return centroids

def get_cluster_weights(data, labels):
    """Get the weights of each cluster based on their size."""
    # Create a list to store the cluster weights
    weights = []

    # Iterate over the unique cluster labels
    for label in np.unique(labels):
        # If the label is not noise
        if label != -1:
            # Get the data points belonging to this cluster
            cluster_data = data[labels == label]
            
            # Compute the weight of the cluster
            weight = len(cluster_data)
            
            # Add the weight to the list
            weights.append(weight)

    # Convert the weights into a NumPy array
    weights = np.array(weights)

    # Normalize the weights to be between 0 and 1
    weights = weights / np.max(weights)
    
    return weights
    
def write_to_json(centroids, weights):
    # Create a list to store the centroid data
    centroid_data = []

    # Iterate over the centroids and their corresponding weights
    for centroid, weight in zip(centroids, weights):
        # Create a dictionary to store the centroid data
        centroid_dict = {
            'x': centroid[1],
            'y': centroid[0],
            'weight': weight
        }
        
        # Add the centroid data to the list
        centroid_data.append(centroid_dict)

    centroid_data.sort(key=lambda x : x["weight"], reverse=True)
    # Write the centroid data to a JSON file
    with open('pop_density_centres.json', 'w') as f:
        json.dump(centroid_data, f, indent=4)

    
def main():
    # Load the image using Pillow
    if len(sys.argv) > 1:
        # Use the specified filename
        filename = sys.argv[1]
    else:
        # must be specified
        print("Please specific population density map name")
        return
        
    image = load_image(filename)
    data = extract_data_points(image)
    labels = cluster_data_points(data)
    centroids = get_cluster_centroids(data, labels)
    weights = get_cluster_weights(data, labels)
    
    # Save as json file
    write_to_json(centroids, weights)

    # Plot the image
    plt.imshow(image, cmap='gray')

    # Plot the clustered points
    plt.scatter(data[:, 1], data[:, 0], c=labels, cmap='rainbow')

    # Plot the cluster centroids
    plt.scatter(centroids[:, 1], centroids[:, 0], c='black', marker='x')

    # Annotate the centroids with their weights
    for i, centroid in enumerate(centroids):
        plt.annotate(f'{weights[i]:.2f}', (centroid[1], centroid[0]), color='white', fontsize=8)
    
    # Show the plot
    plt.show()
    
if __name__ == "__main__":
    main()