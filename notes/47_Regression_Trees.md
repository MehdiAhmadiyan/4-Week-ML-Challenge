# 🌳 Regression with Decision Trees (Predicting a Number)

## 1. Classification vs. Regression Trees
*   **Classification Trees:** Used when the target output ($y$) is a discrete category (e.g., predicting whether an animal is a Cat or Not Cat).
*   **Regression Trees:** Used when the target output ($y$) is a continuous number (e.g., predicting the exact weight of an animal in pounds).

## 2. How Regression Trees Make Predictions
When a regression tree is fully built, how does it predict a number for a brand new test example?

*   The new example flows down the tree from the root node to a specific leaf node based on its features.
*   Instead of predicting a "class", the leaf node predicts a specific number.
*   This number is simply the **average (mean)** of all the target values ($y$) from the training examples that landed in that exact same leaf node during training.
*   *Example:* If a leaf node contains 4 training examples with weights $7.2, 7.6, 8.4$, and $10.2$, the prediction for any new example reaching this leaf will be their average: $8.35$.

## 3. Choosing a Split: Reduction in Variance
In a Classification Tree, we evaluate splits by calculating Information Gain (Reduction in Entropy). In a Regression Tree, we evaluate splits by calculating the **Reduction in Variance**.

*   **Variance:** A statistical measure of how widely a set of numbers varies or spreads out. A highly varied set of numbers has high variance, while a set of very similar numbers has low variance.
*   The goal of the algorithm is to choose the feature that reduces the overall variance the most, grouping animals with similar weights together.

## 4. The Mathematical Formula
To compute the Reduction in Variance, we need the variance of the root node and the weighted average variance of the left and right sub-branches:

$$
\text{Reduction in Variance} = \text{Variance}^\text{root} - \left( w^\text{left} \text{Variance}^\text{left} + w^\text{right} \text{Variance}^\text{right} \right)
$$

## 5. Working Through the Example
Let's evaluate the split for the "Ear Shape" feature using the provided dataset.

*   The variance of all 10 animals at the root node is $20.51$.
*   **Left Branch (Pointy Ears):** Receives 5 out of 10 examples ($w^\text{left} = 5/10$). The variance of these 5 weights is $1.47$.
*   **Right Branch (Floppy Ears):** Receives 5 out of 10 examples ($w^\text{right} = 5/10$). The variance of these 5 weights is $21.87$.

$$
\text{Reduction in Variance} = 20.51 - \left( \frac{5}{10} \times 1.47 + \frac{5}{10} \times 21.87 \right) = 8.84
$$

### The Final Decision
The algorithm computes this reduction for all features:
*   Ear shape Reduction: $8.84$
*   Face shape Reduction: $0.64$
*   Whiskers Reduction: $6.22$

Since **Ear Shape** provides the highest Reduction in Variance ($8.84$), the algorithm selects it as the best feature to split the root node. It then recursively applies this exact same process to build the rest of the regression tree.
