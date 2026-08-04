# 📊 Diagnosing Bias and Variance

## 1. The Golden Rule of Parameters (Key Concept)
Before diagnosing the model, it is crucial to remember exactly how our parameters are derived:
*   **Core Parameters ($\vec{w}, b$):** These are strictly learned and fitted by minimizing the Training Error ($J_{train}$).
*   **Model Architecture ($d$):** Hyperparameters like the degree of the polynomial ($d$) or the size of a neural network are selected by evaluating and minimizing the Cross-Validation Error ($J_{cv}$).

## 2. Diagnosing High Bias (Underfitting)
*   **What it means:** The algorithm has a very strong preconception and fails to even fit the training data well.
*   **The Key Indicator:** The training error is unusually high.
*   **Relationship:** Usually, the cross-validation error will also be high and relatively close to the training error.

**Mathematical Signature:**

$$
J_{train} \text{ is HIGH} \quad \text{and} \quad J_{train} \approx J_{cv}
$$

## 3. Diagnosing High Variance (Overfitting)
*   **What it means:** The algorithm fits the training data perfectly but completely fails to generalize to new, unseen examples.
*   **The Key Indicator:** The cross-validation error is *much* greater than the training error.
*   **Relationship:** The training error might be very low, but the model performs terribly on the cross-validation set.

**Mathematical Signature:**

$$
J_{cv} \gg J_{train}
$$

## 4. The "Just Right" Model
*   **What it means:** The model captures the underlying pattern perfectly without overfitting.
*   **Relationship:** Both errors are low. $J_{train}$ is low, and $J_{cv}$ is also low (and not significantly worse than $J_{train}$).

## 5. The Error Curve vs. Polynomial Degree ($d$)
If you plot $J_{train}$ and $J_{cv}$ as a function of the polynomial degree ($d$), you will see a distinct pattern:

*   **$J_{train}$ Curve:** As the polynomial degree $d$ increases, the model becomes more complex and fits the training data better and better. Therefore, $J_{train}$ continuously **goes down**.
*   **$J_{cv}$ Curve:** This creates a U-shaped curve. 
    *   When $d$ is very small, the model underfits, so $J_{cv}$ is high.
    *   As $d$ increases, $J_{cv}$ comes down and reaches a minimum (the "Just Right" spot).
    *   If $d$ becomes too large, the model overfits, and $J_{cv}$ goes back up.

## 6. High Bias AND High Variance
*   While uncommon in simple 1D linear regression, it is entirely possible for complex models (like Neural Networks) to suffer from both problems simultaneously.
*   **What it means:** The model overfits some parts of the input data while simultaneously underfitting other parts of the data.
*   **The Key Indicator:** The model does poorly on the training set (High Bias), and it does *even worse* on the cross-validation set (High Variance).

**Mathematical Signature:**

$$
J_{train} \text{ is HIGH} \quad \text{and} \quad J_{cv} \gg J_{train}
$$
