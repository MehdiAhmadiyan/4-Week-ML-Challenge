# 🧪 Measuring Purity with Entropy

## 1. What is Entropy?
In the context of Decision Trees, **Entropy** is a mathematical function used to measure the **impurity** of a set of data. 
*   If a set of examples contains only one class (e.g., 100% cats or 100% dogs), it is completely pure, meaning the impurity (entropy) is $0$.
*   If a set is a perfect 50-50 mix of two classes, it is completely impure, meaning the impurity (entropy) is at its maximum value of $1$.

## 2. Defining the Variables
Before writing the formula, we need to define our fractions based on the examples in our node:

*   $p_1$: The fraction of examples that are positive (e.g., Cats).
*   $p_0$: The fraction of examples that are negative (e.g., Dogs). 

Since there are only two classes, we know that $p_0 = 1 - p_1$.

## 3. The Entropy Formula
The entropy function is denoted as $H(p_1)$ and is calculated using the following formula:

$$
H(p_1) = -p_1 \log_2(p_1) - p_0 \log_2(p_0)
$$

Which can also be written entirely in terms of $p_1$:

$$
H(p_1) = -p_1 \log_2(p_1) - (1 - p_1) \log_2(1 - p_1)
$$

### Important Mathematical Notes:
*   **Base 2 Logarithm:** By convention, we compute entropy using $\log_2$ instead of the natural logarithm ($\ln$). This makes the peak of the curve exactly $1.0$, which is easier to interpret.
*   **Handling Zeroes:** If a node is completely pure, $p_1$ or $p_0$ will be $0$. Technically, $\log(0)$ is negative infinity (undefined). However, by convention in this algorithm, we treat **$0 \log(0) = 0$** to make the math work properly.

## 4. The Entropy Curve (Visual Intuition)
If you plot $H(p_1)$ as a curve where the horizontal axis is $p_1$ (from 0 to 1), it forms a bell-like shape:

*   **$p_1 = 0.0 \rightarrow H(p_1) = 0$** (100% Dogs: Completely Pure)
*   **$p_1 = 1/3 \rightarrow H(p_1) \approx 0.92$** (e.g., 2 Cats, 4 Dogs: Highly Impure)
*   **$p_1 = 0.5 \rightarrow H(p_1) = 1$** (50-50 mix: Maximum Impurity)
*   **$p_1 = 5/6 \rightarrow H(p_1) \approx 0.65$** (e.g., 5 Cats, 1 Dog: Somewhat Pure)
*   **$p_1 = 1.0 \rightarrow H(p_1) = 0$** (100% Cats: Completely Pure)

*(Side Note: While Entropy is the primary measure we will use, some open-source packages use a very similar mathematical curve called the **Gini Impurity** or Gini criteria, which works almost identically for building decision trees)*.
