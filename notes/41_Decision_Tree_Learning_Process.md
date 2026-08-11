# 🌳 Decision Tree Learning Process

## 1. The Process of Building a Decision Tree
Building a decision tree is a recursive, step-by-step process of splitting the dataset into smaller and smaller subsets. 

1.  **Start at the Root Node:** Look at all your training examples (e.g., 5 cats and 5 dogs). Choose a specific feature (e.g., "Ear Shape") to split the data.
2.  **Split the Data:** Send the examples with "Pointy Ears" down the left branch and the examples with "Floppy Ears" down the right branch.
3.  **Repeat on the Left Branch:** Look *only* at the subset of data on the left branch. Choose a new feature (e.g., "Face Shape") to split this specific subset again.
4.  **Repeat on the Right Branch:** Do the same for the right branch (e.g., choose the "Whiskers" feature to split that specific subset).
5.  **Create Leaf Nodes:** When a subset contains only one class (e.g., 100% cats or 100% dogs), stop splitting and create a "Leaf Node" that makes the final prediction.

## 2. Key Decision 1: How to choose which feature to split on?
At every single node (starting from the root), the algorithm must decide which feature to use for the split (e.g., Ear Shape, Face Shape, or Whiskers).

*   **The Goal (Maximize Purity):** The algorithm evaluates all available features and chooses the one that results in the highest **Purity** in the resulting sub-branches.
*   **What is Purity?** A completely pure subset contains only one class (e.g., 100% cats or 100% not cats). If a feature splits the data so perfectly that all cats go left and all dogs go right, that is a perfectly pure split.
*   *Note:* In the next video, we will learn how to mathematically calculate this impurity using a concept called **Entropy**.

## 3. Key Decision 2: When do you stop splitting?
If the algorithm keeps splitting the data endlessly, the tree will become massive, unwieldy, and will severely **overfit** the training data. Therefore, the algorithm needs specific criteria to stop splitting:

1.  **100% Purity:** When a node consists of 100% of one single class, stop splitting and turn it into a leaf node.
2.  **Maximum Depth:** You can set a parameter for the maximum depth of the tree (e.g., Depth = 2). The Root Node is at Depth 0. The nodes directly below it are at Depth 1. If splitting a node would exceed the maximum depth, the algorithm stops.
3.  **Minimum Purity Improvement:** If splitting a node barely improves the purity score (the gain is too small), the algorithm stops splitting to keep the tree small.
4.  **Minimum Examples in a Node:** If the number of examples left in a node is too small (e.g., only 3 examples), the algorithm stops splitting to prevent overfitting. It will simply create a leaf node predicting the majority class of those 3 examples.

> **Crucial Advice:** The Decision Tree algorithm might seem messy because it was developed over many years by different researchers adding different rules and criteria. However, all these pieces work incredibly well together, and modern open-source libraries handle most of this complexity automatically.
