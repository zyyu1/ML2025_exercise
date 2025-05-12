import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    distances = np.sum((X[:, np.newaxis] - centroids)**2, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            new_centroids[i] = np.mean(cluster_points, axis=0)
    return new_centroids

def compute_loss(X, centroids, labels):
    loss = 0
    for i in range(centroids.shape[0]):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            loss += np.sum((cluster_points - centroids[i])**2)
    return loss

def kmeans(X, k, max_iters=1000, tol=1e-6):
    centroids = initialize_centroids(X, k)
    loss_history = []

    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        loss = compute_loss(X, new_centroids, labels)
        loss_history.append(loss)

        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids

    return centroids, labels, loss_history
#define functions for k means

def compress_image(img_path, k):
    image = imread(img_path) / 255.0
    w, h, c = image.shape
    X = image.reshape(-1, 3)

    centroids, labels, loss_history = kmeans(X, k)
    compressed_img = centroids[labels].reshape(w, h, 3)
#Map each pixel to its centroids’ RGB value to form the compressed image.

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image)
    axs[0].set_title("Original")
    axs[0].axis('off')
    
    axs[1].imshow(compressed_img)
    axs[1].set_title(f"Compressed (k={k})")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()
# Plot original and compressed image

    plt.figure()
    plt.plot(range(len(loss_history)), loss_history, 'bo-')
    plt.title("K-means Loss Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
# Plot loss curve

    return centroids, labels, loss_history

for k in [2, 4, 32]:
    print(f"\nRunning K-means for K={k}...")
    compress_image("car.jpe", k)
# Run for K = 2, 4, 32