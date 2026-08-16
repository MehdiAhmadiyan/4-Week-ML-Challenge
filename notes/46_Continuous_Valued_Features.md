# 📏 Splitting on Continuous-Valued Features

## 1. What is a Continuous Feature?
*   Until now, we have only worked with discrete, categorical features (e.g., Ear shape being pointy, floppy, or oval). 
*   A continuous feature is one that can be any number, such as the weight of an animal in pounds (e.g., 7.2, 8.8, 15).
*   To use a continuous feature in a Decision Tree, we cannot create a separate branch for every single number. Instead, we split the data based on a **threshold** (e.g., Weight $\le 8$ lbs).

## 2. How the Algorithm Finds the Best Threshold
If the feature is continuous, how does the algorithm know which threshold number to pick? It follows a systematic process:

1.  **Sort the data:** It takes all the training examples and sorts them in order based on the value of the continuous feature (e.g., from lightest to heaviest).
2.  **Find the midpoints:** It takes the midpoints between every two adjacent sorted values to use as candidate thresholds. For example, if you have 10 training examples, the algorithm will test 9 different possible thresholds.
3.  **Calculate Information Gain:** For *every single candidate threshold*, the algorithm splits the data into two sets (Left: $\le$ threshold, Right: $>$ threshold) and calculates the Information Gain.
4.  **Pick the best one:** It selects the threshold that yields the highest Information Gain.

## 3. Example Calculations (Weight of Cats vs. Dogs)
Let's look at the root node with 10 animals (5 cats, 5 dogs), where the initial entropy is $H(0.5) = 1.0$. The algorithm tests multiple thresholds:

**Test A: Threshold $\le 8$ lbs**
*   **Left Subset ($\le 8$):** 2 examples (2 cats, 0 dogs).
*   **Right Subset ($> 8$):** 8 examples (3 cats, 5 dogs).

$$
\text{Information Gain} = H(0.5) - \left( \frac{2}{10} H\left(\frac{2}{2}\right) + \frac{8}{10} H\left(\frac{3}{8}\right) \right) = 0.24
$$

**Test B: Threshold $\le 9$ lbs**
*   **Left Subset ($\le 9$):** 4 examples (4 cats, 0 dogs).
*   **Right Subset ($> 9$):** 6 examples (1 cat, 5 dogs).

$$
\text{Information Gain} = H(0.5) - \left( \frac{4}{10} H\left(\frac{4}{4}\right) + \frac{6}{10} H\left(\frac{1}{6}\right) \right) = 0.61
$$

**Test C: Threshold $\le 13$ lbs**
*   **Left Subset ($\le 13$):** 7 examples (5 cats, 2 dogs).
*   **Right Subset ($> 13$):** 3 examples (0 cats, 3 dogs).

$$
\text{Information Gain} = H(0.5) - \left( \frac{7}{10} H\left(\frac{5}{7}\right) + \frac{3}{10} H\left(\frac{0}{3}\right) \right) = 0.40
$$

## 4. The Final Decision
*   After testing the different thresholds for the continuous feature, the algorithm identifies that setting the threshold at **9 lbs** produces the best Information Gain ($0.61$).
*   Finally, the algorithm compares this $0.61$ score with the Information Gain of all the *other* available features (like Ear shape, Face shape, etc.). 
*   If $0.61$ is higher than the Information Gain of any other feature, the algorithm will choose to split the node using the continuous feature "Weight $\le 9$".
