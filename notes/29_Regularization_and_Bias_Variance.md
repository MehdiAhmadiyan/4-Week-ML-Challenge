# ⚖️ Regularization and Bias/Variance

## 1. The Impact of Lambda ($\lambda$) on the Model
Just like the degree of polynomial ($d$) affects bias and variance, the choice of the regularization parameter ($\lambda$) has a massive impact on the model's performance.

*   **Large $\lambda$ (e.g., $\lambda = 10,000$):**
    *   The algorithm is highly motivated to keep the parameters $w_j$ very close to zero.
    *   The model flattens out (e.g., $f(x) \approx b$) and **underfits** the data.
    *   **Result:** High Bias.
*   **Small $\lambda$ (e.g., $\lambda = 0$):**
    *   There is zero (or very little) regularization penalty, allowing the parameters to grow large.
    *   The model becomes overly complex and **overfits** the data.
    *   **Result:** High Variance.
*   **Intermediate $\lambda$ (e.g., $\lambda = 0.08$):**
    *   The penalty is perfectly balanced. It keeps parameters small enough to avoid overfitting, but large enough to fit the data properly.
    *   **Result:** "Just Right" (Low Bias, Low Variance).

## 2. Choosing the Best $\lambda$ using Cross-Validation
Just as we used the Cross-Validation set to choose the polynomial degree ($d$), we use the exact same procedure to choose $\lambda$:

1.  **Define a list of $\lambda$ values to try:** Start at 0, and incrementally double the value (e.g., $\lambda = 0, 0.01, 0.02, 0.04, 0.08 \dots$ up to around 10).
2.  **Find $\vec{w},b$ using $J_{train}$:** For *each* value of $\lambda$, minimize the regularized cost function on the Training Set to find the specific parameters $\vec{w}$ and $b$ for that model.
3.  **Evaluate using $J_{cv}$:** Take those trained parameters and evaluate them on the Cross-Validation Set. Record the $J_{cv}$ error for each model.
4.  **Select the Best Model:** Pick the value of $\lambda$ that produced the lowest Cross-Validation error ($J_{cv}$).
5.  **Report the Test Error:** Finally, evaluate the single chosen model on the hidden Test Set ($J_{test}$) to report the unbiased generalization error.

## 3. The Error Curves vs. Lambda ($\lambda$)
If you plot $J_{train}$ and $J_{cv}$ as a function of $\lambda$, the curves look like **mirror images** of the curves plotted against polynomial degree ($d$):

*   **The $J_{train}$ Curve:**
    *   When $\lambda$ is small, the model fits the training data perfectly, so $J_{train}$ is very low.
    *   As $\lambda$ increases, the algorithm focuses more on shrinking the parameters rather than fitting the data, so $J_{train}$ continuously **goes up**.
*   **The $J_{cv}$ Curve (U-Shape):**
    *   On the left (Small $\lambda$): The model overfits (High Variance), so $J_{cv}$ is high.
    *   In the middle (Intermediate $\lambda$): The model generalizes well, so $J_{cv}$ dips to its lowest point (The "Just Right" zone).
    *   On the right (Large $\lambda$): The model underfits (High Bias), so $J_{cv}$ goes back up.
