# 🧩 Putting It Together: The Decision Tree Learning Algorithm

## 1. The Complete Process
Building a decision tree involves bringing together Information Gain and stopping criteria into a single, cohesive process. 

Here is the step-by-step procedure:
1.  **Start at the Root:** Begin with all the training examples grouped together at the root node.
2.  **Calculate Information Gain:** For *all* available features, calculate the Information Gain (the reduction in entropy).
3.  **Pick the Best Feature:** Choose the feature that yields the highest Information Gain.
4.  **Split the Data:** Divide the dataset according to the selected feature and send the examples down the newly created left and right branches.
5.  **Repeat (Recursion):** Treat the left sub-branch as if it were a brand new root node with a smaller dataset. Repeat steps 2-4. Then, do the exact same thing for the right sub-branch.
6.  **Stop:** Keep repeating this splitting process on every new branch until a stopping criterion is met.

## 2. The Stopping Criteria
The algorithm needs rules to know when to stop splitting, otherwise, it will build a massive tree that perfectly memorizes the training data (severe overfitting). The algorithm stops when *any* of these conditions are met:

*   **100% Purity:** When a node consists entirely of one single class (Entropy = 0).
*   **Maximum Depth:** When splitting a node would cause the tree to exceed a pre-set maximum depth limit. (A larger maximum depth allows the tree to learn more complex patterns, similar to a larger neural network, but increases the risk of overfitting).
*   **Information Gain is Too Small:** If splitting a node results in an Information Gain that is less than a specific threshold, the algorithm decides it's not worth the effort and stops splitting to avoid unnecessarily growing the tree.
*   **Too Few Examples:** When the number of examples left in a node falls below a specific threshold (e.g., only 3 examples left). 

When a stopping criterion is met, the algorithm turns that specific node into a **Leaf Node** that makes a final prediction based on the majority class of the remaining examples.

## 3. The Concept of "Recursive Algorithm"
*   If you look at how the tree is built, once the root node is split, the algorithm builds the left sub-tree by acting as if it's training a completely new, smaller decision tree from scratch. It then does the same for the right sub-tree.
*   In computer science, a function that solves a larger problem by repeatedly calling itself to solve smaller versions of the exact same problem is called a **Recursive Algorithm**. 
*   If you ever look at the source code of a decision tree library, you will see recursion in action. However, modern open-source libraries handle this complexity perfectly behind the scenes, so you don't need to write the recursive code from scratch.
