# 🎯 K-Means Clustering: Intuition

## 1. What is K-Means?
K-Means is an unsupervised learning algorithm that automatically groups an unlabeled dataset into a specific number of clusters. 
The "centers" of these clusters are called **Cluster Centroids**.

## 2. The Setup (Initialization)
*   Imagine you have a dataset of 30 unlabeled points scattered on a graph.
*   Before doing any math, you must decide how many clusters you want to find (e.g., 2 clusters).
*   **Initial Guess:** The algorithm starts by randomly placing 2 points on the graph (e.g., a Red Cross and a Blue Cross). These act as the initial, random guesses for where the cluster centroids might be.

## 3. The Two-Step Loop
Once the initial random centroids are placed, the K-Means algorithm repeatedly performs two distinct steps:

### Step 1: Assign points to cluster centroids
*   The algorithm goes through every single data point in the dataset one by one.
*   For each point, it measures the distance to the Red Cross and the distance to the Blue Cross.
*   It then "assigns" (or colors) the point based on whichever centroid is closer. 
*   *Result:* All points closer to the red centroid become the "Red Cluster", and all points closer to the blue centroid become the "Blue Cluster".

### Step 2: Recompute (Move) the centroids
*   Now, the algorithm looks *only* at the points assigned to the Red Cluster. It calculates the **average (mean) location** of all these red dots.
*   It picks up the Red Cross and moves it to this newly calculated average location.
*   It does the exact same thing for the Blue Cluster, moving the Blue Cross to the average location of all the blue dots.

## 4. Convergence (When does it stop?)
*   After Step 2, the centroids have moved to new locations. Because they moved, some dots that were previously closer to the red centroid might now actually be closer to the blue centroid.
*   Therefore, the algorithm goes back to Step 1 and re-evaluates all the distances, changing the colors of the points if necessary. Then it does Step 2 again to move the centroids.
*   **Convergence:** The algorithm keeps repeating Step 1 and Step 2 over and over again until it reaches a point where the colors of the dots stop changing and the centroids stop moving. 
*   When absolutely nothing changes after a full loop, the algorithm has **converged**, meaning it has successfully found the final clusters!
