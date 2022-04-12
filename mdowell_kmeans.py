"""
Meghan Dowell
11 April 2022
CS 7375 Artificial Intelligence
Assignment 2
"""

import random
import matplotlib.pyplot as plt
import numpy
import numpy as np


# info from https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
def euclidean_distance(a, b):
    # returns the euclidian distance between the given points - 2D
    return numpy.linalg.norm(a-b)


# Step 1
def closestCentroid(x, centroids):
    assignments = []  # list of which centroids to assign to each point in x
    for i in x:  # loop through each point in x
        # distance between one data point and centroids
        distance = []  # record distances between the centroid and the points
        for j in centroids:  # for each centroid
            # add the distance between the point and the centroid to distances list
            distance.append(euclidean_distance(i, j))
        # assign each data point to the cluster with closest centroid
        assignments.append(np.argmin(distance))
    return np.array(assignments)  # return the assignments


# Step 2
def updateCentroid(x, clusters, k):
    new_centroids = []  # list of updated centroids
    for c in range(k):  # for each centroid
        # Update the cluster centroid with the average of all points in this cluster
        points = x[clusters == c]  # get the points that are assigned to this cluster
        cluster_mean_x = np.array([t[0] for t in points]).mean()  # get the average of x values of each point in this cluster
        cluster_mean_y = np.array([t[1] for t in points]).mean()  # get the average of y values of each point in this cluster
        new_centroids.append([cluster_mean_x, cluster_mean_y])  # add the average x and y values of this cluster as a point to be the new centroid
    return new_centroids  # return the updated centroids


# 2-d kmeans
def kmeans(x, k):
    # get k unique numbers in range [0, length of x)
    indices = random.sample(range(0, len(x)), k)

    # use the k points at the random indices for the starting centroids
    centroids = []
    for i in range(k):  # loop for number of centroids
        centroids.append(x[indices[i]])  # add the point at that index to centroids list
    centroids = np.array(centroids)
    print('Initialized centroids: {}'.format(centroids))

    for i in range(10):  # iterate 10 times
        clusters = closestCentroid(x, centroids)  # get the new clusters
        centroids = updateCentroid(x, clusters, k)  # get the new centroids
        print('Iteration: {}, Centroids: {}'.format(i, centroids))
    return centroids  # return the final centroids


# add clusters to scatter plot
def visualizeClusters(x, clusters, k):
    for c in range(k):  # for each centroid
        points = x[clusters == c]  # get the points associated with the centroid
        #  plot the points on the graph with a random RGB
        plt.scatter(points[:, 0], points[:, 1], s=150, color=numpy.random.rand(3,))


# data array
X = np.array([[2, 4],
              [1.7, 2.8],
              [7, 8],
              [8.6, 8],
              [3.4, 1.5],
              [9, 11]])

# X = np.array([[4, 2],
#               [3.6, 1.7],
#               [1, 2.8],
#               [9, 9],
#               [12, 1.9],
#               [6, 8],
#               [9.2, 2.5]])

K = 2  # number of clusters/centroids
finalCentroids = np.array(kmeans(X, K))  # get the final centroids
finalClusters = closestCentroid(X, finalCentroids)  # get the final clusters
visualizeClusters(X, finalClusters, K)   # visualize the clusters

# plot the centroids in black and 1/2 size of other points
plt.scatter(finalCentroids[:, 0], finalCentroids[:, 1], s=75, c="black")
plt.show()  # show the scatter plot
