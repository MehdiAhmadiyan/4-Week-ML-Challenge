# 🎯 Model Selection and Cross-Validation

## 1. The Flaw in the "Train/Test" Only Method
* Suppose you want to choose the best model (e.g., deciding whether to use a 1st-degree, 2nd-degree, or 10th-degree polynomial).
* If you fit parameters $\vec{w},b$ using the Training Set, and then use the Test Set to choose the best polynomial degree ($d$), the Test Set error ($J_{test}$) will be **overly optimistic**.
* Why? Because you essentially used the Test Set to fit an extra parameter (the degree $d$). Therefore, the Test Set is no longer a fair, independent judge of how well the model generalizes to brand new data.

## 2. The Solution: Three Data Subsets
To properly carry out "Model Selection" (choosing among different models), you should split your entire dataset into **three** separate subsets:

1.  **Training Set (e.g., 60%):** Used strictly to train the model and find the parameters. Number of examples: $m_{train}$.
2.  **Cross-Validation Set (e.g., 20%):** Used to check the validity of different models and choose the best one. Number of examples: $m_{cv}$.
    *   *Note:* This is also commonly called the **Validation Set**, the **Development Set**, or the **Dev Set** for short.
3.  **Test Set (e.g., 20%):** Used *only at the very end* to report an unbiased estimate of the final model's generalization error. Number of examples: $m_{test}$.

## 3. Calculating Errors (Without Regularization)
Just like before, none of these error formulas include the regularization term.

**Training Error ($J_{train}$):**

$$
J_{train}(\vec{w},b) = \frac{1}{2m_{train}} \sum_{i=1}^{m_{train}} (f_{\vec{w},b}(\vec{x}_{train}^{(i)}) - y_{train}^{(i)})^2
$$

**Cross-Validation Error ($J_{cv}$):**

$$
J_{cv}(\vec{w},b) = \frac{1}{2m_{cv}} \sum_{i=1}^{m_{cv}} (f_{\vec{w},b}(\vec{x}_{cv}^{(i)}) - y_{cv}^{(i)})^2
$$

**Test Error ($J_{test}$):**

$$
J_{test}(\vec{w},b) = \frac{1}{2m_{test}} \sum_{i=1}^{m_{test}} (f_{\vec{w},b}(\vec{x}_{test}^{(i)}) - y_{test}^{(i)})^2
$$

## 4. The Proper Procedure for Model Selection
This is the industry-standard best practice for choosing a model architecture (like polynomial degree $d$ or neural network size):

1.  **Find $\vec{w}$ and $b$ using $J_{train}$:** For every single model you are considering (e.g., $d=1$ up to $d=10$), use the Training Set ($J_{train}$) to fit and find the parameters $\vec{w}$ and $b$.
2.  **Choose the Model using $J_{cv}$:** Evaluate all those trained models on the Cross-Validation Set. Look at their $J_{cv}$ scores and pick the model that has the lowest Cross-Validation error.
3.  **Evaluate using $J_{test}$:** Finally, take the single model you just selected and evaluate it on the Test Set ($J_{test}$). Because the Test Set was never used to fit $\vec{w}, b$, or $d$, it provides a fair and accurate estimate of generalization error.

> **Golden Rule:** Make all decisions about your model (parameters, architecture, polynomial degree) using *only* the Training Set and Cross-Validation Set. Do not look at the Test Set at all during the decision-making process.
