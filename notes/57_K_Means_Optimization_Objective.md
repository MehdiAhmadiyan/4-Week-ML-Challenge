# 🎯 K-Means Optimization Objective (Cost Function)

## 1. Mathematical Notation
To understand what K-Means is mathematically trying to achieve, we first need to define our variables clearly:
*   $c^{(i)}$: The index of the cluster ($1, 2, \dots, K$) to which training example $x^{(i)}$ is currently assigned.
*   $\mu_k$: The exact location of cluster centroid $k$.
*   $\mu_{c^{(i)}}$: The location of the specific cluster centroid to which example $x^{(i)}$ has been assigned. 

*(For example, if example number 10 is assigned to cluster 2, then $c^{(10)} = 2$, and the location of its centroid is $\mu_2$)*.

## 2. The Cost Function (Distortion)
In supervised learning, algorithms optimize a cost function (like Mean Squared Error). K-Means, despite being an unsupervised algorithm, also optimizes a specific cost function. 

This cost function is often called the **Distortion Function** or simply **Distortion**, denoted by $J$.

$$
J(c^{(1)}, \dots, c^{(m)}, \mu_1, \dots, \mu_K) = \frac{1}{m} \sum_{i=1}^{m} ||x^{(i)} - \mu_{c^{(i)}}||^2
$$

**What does this formula mean?**
The cost function computes the **average squared distance** between every single training example ($x^{(i)}$) and the specific cluster centroid to which it is assigned ($\mu_{c^{(i)}}$). The ultimate goal of the K-Means algorithm is to minimize this cost function $J$.

## 3. How K-Means Minimizes the Cost Function
The two steps of the K-Means algorithm (which we learned previously) are perfectly designed to minimize this exact cost function:

### Step 1: Assign points to closest centroids
*   In this step, the algorithm holds the locations of the centroids ($\mu$) completely fixed.
*   It updates the cluster assignments ($c$) for every point.
*   By assigning a point to its *closest* centroid, the distance $||x^{(i)} - \mu_{c^{(i)}}||^2$ is made as small as mathematically possible for that specific point.

### Step 2: Move the centroids
*   In this step, the algorithm holds the cluster assignments ($c$) completely fixed.
*   It updates the locations of the centroids ($\mu$).
*   Mathematically, taking the **mean (average)** of a set of points is the exact operation that minimizes the sum of squared distances to those points. Therefore, moving the centroid to the average location optimally minimizes the cost function for that cluster.

## 4. Convergence and Debugging
Knowing about the Distortion function gives us two major practical benefits:

1.  **Guaranteed Convergence:** Because every single step of K-Means (both assigning points and moving centroids) is strictly designed to reduce the cost function, the value of $J$ should go down (or stay exactly the same) on every single iteration.
2.  **Debugging Code:** If you plot the cost function after every iteration and see the value of $J$ *increasing*, it means there is a bug in your code. $J$ must never go up.
3.  **Knowing When to Stop:** If the cost function stops going down (or goes down incredibly slowly), you know the algorithm has converged, and you can stop running it.
