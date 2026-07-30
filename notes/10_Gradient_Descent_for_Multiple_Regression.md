# 📈 Gradient Descent for Multiple Linear Regression

## 1. Updating the Notation (Vector Form)
To handle multiple linear regression cleanly, we transition from writing out individual parameters ($w_1, w_2, ..., w_n$) to using vector notation.
*   **Parameters:** Instead of thinking of $w_1$ to $w_n$ as separate numbers, we collect them into a single vector $\vec{w}$ of length $n$. The parameter $b$ remains a single scalar number.
*   **The Model:** The model is now written using the dot product: $f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$.
*   **The Cost Function:** Instead of writing $J(w_1, ..., w_n, b)$, we simply write it as $J(\vec{w}, b)$. It takes the vector $\vec{w}$ and the number $b$ as inputs and returns a single number representing the cost.

## 2. Gradient Descent with Multiple Features
When transitioning from one feature to $n$ features, the gradient descent update rules adjust to update every single parameter in the vector $\vec{w}$.

*   You must repeatedly update the parameters until convergence:
    *   $w_j = w_j - \alpha \frac{\partial}{\partial w_j} J(\vec{w},b)$ (for $j = 1 \dots n$)
    *   $b = b - \alpha \frac{\partial}{\partial b} J(\vec{w},b)$
*   **The specific derivative formula for each $w_j$ is:**
    $$w_j = w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})x_j^{(i)}$$
*   Notice that the formula looks nearly identical to the univariate version, except now $\vec{x}$ and $\vec{w}$ are vectors, and we multiply the error term specifically by $x_j^{(i)}$ (the $j^{th}$ feature of the $i^{th}$ training example).
*   As always, you must **simultaneously update** all parameters ($w_1$ through $w_n$ and $b$) at the same time.

## 3. The Normal Equation (An Alternative Method)
While gradient descent is the standard iterative algorithm for minimizing the cost function $J$, there is an alternative mathematical approach called the **Normal Equation**.

*   **How it works:** It uses an advanced linear algebra library to solve for $\vec{w}$ and $b$ exactly in one single step, without needing any iterations or a learning rate $\alpha$.
*   **Disadvantages:**
    1.  It **only** works for linear regression. It does not generalize to other algorithms like logistic regression or neural networks.
    2.  It becomes computationally **very slow** if the number of features ($n$) is large.
*   **Practical Usage:** Almost no machine learning practitioners implement the normal equation themselves. However, if you use a mature ML library (like Scikit-Learn) and call its linear regression function, there is a chance it might use this method in the backend to solve for the parameters. 
*   **Why it matters:** It is important to know what the "Normal Equation" refers to, especially if the term comes up in a job interview. For almost all implementations, however, gradient descent is the better and more versatile way to get the job done.
