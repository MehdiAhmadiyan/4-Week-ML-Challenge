# ⚖️ Trading Off Precision and Recall

## 1. The Trade-off
In an ideal world, we want a learning algorithm with both high precision and high recall. However, in practice, increasing one almost always decreases the other. We manage this trade-off by adjusting the **decision threshold** of our logistic regression model.

Usually, we predict $y=1$ if $f_{\vec{w},b}(\vec{x}) \ge 0.5$. We can change this $0.5$ threshold depending on our goal:

### Scenario A: High Precision, Lower Recall
*   **Goal:** Predict $y=1$ (e.g., patient has the rare disease) *only* if we are very confident. (Use this when the treatment is invasive, painful, or highly expensive).
*   **Action:** Raise the threshold (e.g., to $0.7$ or $0.9$).
*   **Result:** Precision increases (when you say someone is sick, you are almost certainly right). Recall decreases (you will miss some sick patients because you are being too cautious).

### Scenario B: Lower Precision, Higher Recall
*   **Goal:** Avoid missing too many cases of the rare disease. When in doubt, predict $y=1$. (Use this when leaving the disease untreated is much worse than a false alarm).
*   **Action:** Lower the threshold (e.g., to $0.3$ or $0.1$).
*   **Result:** Recall increases (you successfully detect almost all sick patients). Precision decreases (you will cause many false alarms for healthy patients).

## 2. Choosing the Best Algorithm: The F1 Score
Suppose you have trained 3 different algorithms, and they have different Precision (P) and Recall (R) scores. How do you objectively choose the best one?

*   **Why Average doesn't work:** Taking the simple average $\frac{P+R}{2}$ is a bad idea. An algorithm that simply predicts $y=1$ all the time will have a Recall of 1.0 and a Precision near 0, but its average might still look artificially high (e.g., 0.5).
*   **The Solution ($F_1$ Score):** The $F_1$ score (also known mathematically as the Harmonic Mean) is a better way to combine Precision and Recall. It heavily penalizes extreme values, meaning an algorithm will only get a high $F_1$ score if *both* Precision and Recall are decently high.

**The Formula:**

$$
F_1 \text{ Score} = \frac{1}{\frac{1}{2}(\frac{1}{P} + \frac{1}{R})} = 2 \frac{P \times R}{P + R}
$$

**Example Comparison:**
*   Algorithm 1 ($P=0.5, R=0.4$) $\rightarrow$ $F_1 = 0.444$ **(Winner!)**
*   Algorithm 2 ($P=0.7, R=0.1$) $\rightarrow$ $F_1 = 0.175$
*   Algorithm 3 ($P=0.02, R=1.0$) $\rightarrow$ $F_1 = 0.039$

By calculating the $F_1$ score, you can automatically evaluate and select the best algorithm for skewed datasets without having to manually guess which trade-off is mathematically optimal.
