# 🧮 The K-Means Algorithm (Mathematical Details)

## 1. Defining the Variables
Before writing the algorithm, we need to define the mathematical notations used in K-Means:
*   $K$: The total number of clusters we want to find.
*   $m$: The total number of training examples.
*   $x^{(i)}$: A specific training example (an $n$-dimensional vector).
*   $\mu_k$: The exact location (coordinates) of cluster centroid $k$.
*   $c^{(i)}$: The index of the cluster (from $1$ to $K$) to which the training example $x^{(i)}$ is currently assigned.

## 2. The Algorithm Steps
The K-Means algorithm starts by randomly initializing the $K$ cluster centroids ($\mu_1, \mu_2, \dots, \mu_K$). After this initial random placement, it repeatedly executes a loop with two main steps:

### Step 1: Assign points to cluster centroids
*   The algorithm loops through every single training example from $i = 1$ to $m$.
*   For each example $x^{(i)}$, it calculates the distance to all $K$ centroids.
*   It sets $c^{(i)}$ to the index of the centroid that is closest to $x^{(i)}$.

To find the closest centroid, the algorithm minimizes the squared distance (L2 norm) between the data point and the centroid:

$$
\text{Minimize: } ||x^{(i)} - \mu_k||^2
$$

### Step 2: Move cluster centroids
*   Next, the algorithm loops through each of the clusters from $k = 1$ to $K$.
*   It updates the location of $\mu_k$ by calculating the average (mean) of all the data points that were just assigned to cluster $k$.

For example, if training examples $x^{(1)}, x^{(5)}, x^{(6)}, \text{and } x^{(10)}$ are assigned to cluster $1$, the new centroid location is calculated as:

$$
\mu_1 = \frac{1}{4} \left[ x^{(1)} + x^{(5)} + x^{(6)} + x^{(10)} \right]
$$

## 3. Handling a "Corner Case" (Empty Clusters)
*   **The Problem:** What happens if the algorithm moves the centroids in such a way that a specific cluster ends up with **zero** points assigned to it? If this happens, trying to calculate the average of zero points is mathematically impossible (not well-defined).
*   **Primary Solution:** The most common approach is to simply eliminate that cluster entirely, meaning you will end up with $K-1$ clusters.
*   **Alternative Solution:** If your application absolutely requires exactly $K$ clusters, you can randomly reinitialize that empty cluster's centroid to a new location on the graph and continue the algorithm.

## 4. K-Means on "Not Well-Separated" Clusters
*   K-Means is incredibly useful even when the data does not form obvious, well-separated groups.
*   **T-Shirt Sizing Example:** Imagine plotting the heights and weights of your customers. The data will look like one continuous blob, not separate groups. 
*   However, if you run K-Means with $K=3$, the algorithm will still systematically divide this continuous blob into three distinct sections. 
*   These three centroids can then be used to perfectly design the dimensions for Small, Medium, and Large (S, M, L) t-shirts based on representative customer averages.
