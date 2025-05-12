from sklearn.datasets import load_iris
# Load dataset  

data = load_iris()  
X = data.data

#test
#print("Data shape:", X.shape)

import numpy as np
import matplotlib.pyplot as plt

def compute_distances(X, centroids) :
    distances = np.sum((X[:, np.newaxis] - centroids)**2, axis=2)
    return distances
#define calculate distance funktion,For each data point, calculate the Euclidean distance  to each centroid.

def assign_cluster(distance) : 
    return np.argmin(distance, axis=1)
#define assign clusters funktion,Assign each data point to the nearest centroid’s cluster  based on the calculated distances.

def update_centroids(X, labels, k):
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            new_centroids[i] = np.mean(cluster_points, axis=0)
    return new_centroids
#define Recalculate the centroids for each cluster by taking the  mean of all data points assigned to that cluster

def compute_loss(X, centroids, labels):
    loss = 0
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            loss += np.sum((cluster_points - centroids[i])**2)
    return loss
#define loss funktion

k = 3
max_iters = 100
tolerance = 1e-6
runs = 10

def kmeans(X, k, max_iters=100, tol=1e-6):
    initial_indices_centroids = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[initial_indices_centroids]
    loss_history = []

    for _ in range(max_iters):
        distances = compute_distances(X, centroids)
        labels = assign_cluster(distances)
        new_centroids = update_centroids(X, labels, k)
        loss = compute_loss(X, new_centroids, labels)
        loss_history.append(loss)
        
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        
        centroids = new_centroids

    return centroids, labels, loss_history[-1], loss_history
#define Convergence Check:Repeat steps 2-4 until the centroids update change is  not greater than 1 × 10−6.

best_loss = float('inf')
best_centroids = None
best_labels = None
best_loss_history = None

for run in range(runs):
    centroids, labels, loss, loss_history = kmeans(X, k, max_iters, tolerance)

    #test
    #print(f"Run {run + 1} Loss: {loss:.4f}")

    if loss < best_loss:
        best_loss = loss
        best_centroids = centroids
        best_labels = labels
        best_loss_history = loss_history
#Multiple Runs: Re-run the K-means algorithm 10 times with different initial  centroids to find the clustering result with the minimum loss.

print("\nBest Centroids:\n", best_centroids)
print("\nbest_labels", best_labels)
print("Sum of Squared Errors (SSE):", best_loss)
#output

plt.plot(range(len(loss_history)), loss_history, 'bo-')

plt.xlabel('Iteration')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('The minimum loss among 10 iterations')

plt.grid(True)
plt.tight_layout()
plt.show()
#plot