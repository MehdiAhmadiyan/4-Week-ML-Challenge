# 📈 Choosing a Split: Information Gain

## 1. The Core Concept
When building a decision tree, the algorithm needs to decide which feature to use to split the data at any given node. The goal is to choose the feature that **reduces entropy (impurity) the most**. 

In decision tree learning, this reduction in entropy is mathematically measured using a concept called **Information Gain**. The algorithm calculates the Information Gain for every possible feature and simply chooses the one with the highest value.

## 2. Why use a "Weighted" Average?
When we split a node, the data divides into a left branch and a right branch. Each branch will have its own entropy. We cannot simply average these two entropies together. 

Why? Because if one branch contains $1,000$ examples and the other branch contains only $2$ examples, the purity of the branch with $1,000$ examples is vastly more important. Therefore, we must calculate a **weighted average** based on the fraction of total examples that went to each branch.

## 3. The Information Gain Formula
To compute the Information Gain, we define the following variables:

*   $p_1^\text{root}$: The fraction of positive examples (e.g., cats) in the starting/parent node.
*   $w^\text{left}$: The fraction of total examples that went to the left branch.
*   $p_1^\text{left}$: The fraction of positive examples in the left branch.
*   $w^\text{right}$: The fraction of total examples that went to the right branch.
*   $p_1^\text{right}$: The fraction of positive examples in the right branch.

The formula for Information Gain is the entropy of the root node minus the weighted average entropy of the left and right sub-branches:

$$
\text{Information Gain} = H(p_1^\text{root}) - \left( w^\text{left} H(p_1^\text{left}) + w^\text{right} H(p_1^\text{right}) \right)
$$

## 4. Working Through the Example
Let's look at the Root Node from our previous example. We start with 10 animals (5 cats, 5 dogs). 
*   $p_1^\text{root} = 5/10 = 0.5$.
*   The entropy at the root node is $H(0.5) = 1.0$ (Maximum Impurity).

The algorithm calculates the Information Gain for all three available features:

### Option A: Split on Ear Shape
*   **Left Branch (Pointy):** Receives 5 out of 10 examples ($w^\text{left} = 5/10$). It has 4 cats, so $p_1^\text{left} = 4/5 = 0.8$. The entropy $H(0.8) \approx 0.72$.
*   **Right Branch (Floppy):** Receives 5 out of 10 examples ($w^\text{right} = 5/10$). It has 1 cat, so $p_1^\text{right} = 1/5 = 0.2$. The entropy $H(0.2) \approx 0.72$.
*   **Information Gain:** $1.0 - ((5/10 \times 0.72) + (5/10 \times 0.72)) = 0.28$.

### Option B: Split on Face Shape
*   Using the exact same formula, the Information Gain turns out to be **0.03**.

### Option C: Split on Whiskers
*   Using the exact same formula, the Information Gain turns out to be **0.12**.

### The Decision
The algorithm compares the Information Gain values ($0.28$, $0.03$, and $0.12$). Since **Ear Shape** provides the highest Information Gain ($0.28$), it successfully reduces the entropy the most. The algorithm chooses Ear Shape as the feature for the root node.

## 5. Why Calculate Information Gain Instead of Just Entropy?
You might wonder: why subtract the weighted entropy from the root node's entropy? Why not just pick the split with the lowest weighted entropy?

*   **Stopping Criteria:** Information Gain is essentially the *amount of progress* the algorithm made by splitting. 
*   If the Information Gain is extremely small (e.g., below a certain threshold), the algorithm realizes that splitting the node further doesn't significantly improve purity. 
*   It will then decide to **stop splitting** to prevent the tree from becoming unnecessarily large and risking overfitting.
