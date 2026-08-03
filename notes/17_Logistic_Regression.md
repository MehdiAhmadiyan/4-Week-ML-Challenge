# 📈 Logistic Regression

## 1. What is Logistic Regression?
* Logistic regression is probably the single most widely used classification algorithm in the world.
* Instead of fitting a straight line, logistic regression ends up fitting an S-shaped curve to the dataset.
* While the true output label $y$ is strictly $0$ or $1$ in binary classification, the algorithm itself outputs a decimal number between $0$ and $1$ to represent a likelihood or probability.

## 2. The Sigmoid (Logistic) Function
To ensure the output stays strictly between $0$ and $1$, the algorithm uses an important mathematical function called the **Sigmoid function** (also referred to as the logistic function).

*   **The Formula:** 
    $$g(z) = \frac{1}{1 + e^{-z}}$$
    *(Here, $e$ is a mathematical constant approximately equal to $2.7$)*.
*   **Key Properties of the Sigmoid Function:**
    *   When $z$ is a very large positive number (e.g., $100$), $e^{-z}$ becomes a tiny fraction, making the denominator very close to $1$, so $g(z)$ becomes very close to $1$.
    *   When $z$ is a very large negative number, $e^{-z}$ becomes a giant number, making the fraction $1$ over a giant number, so $g(z)$ becomes very close to $0$.
    *   When $z = 0$, $e^{-0}$ equals $1$, making $g(z) = \frac{1}{1+1} = 0.5$. This means the curve crosses the vertical axis exactly at $0.5$.

## 3. Building the Logistic Regression Model
The logistic regression model is built by combining linear regression with the Sigmoid function in two simple steps:

1.  **Step 1 (Linear Equation):** Calculate the straight-line equation $z = \vec{w} \cdot \vec{x} + b$.
2.  **Step 2 (Sigmoid Transformation):** Pass that calculated value of $z$ into the Sigmoid function $g(z)$.

*   **The Final Model Equation:**
    $$f_{\vec{w},b}(\vec{x}) = g(\vec{w} \cdot \vec{x} + b) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$$

## 4. Interpreting the Output (Probability)
* The output of the logistic regression model should be interpreted as the **probability** that the true label $y$ will be equal to $1$ given a certain input $\vec{x}$.
* For example, if a patient has a tumor size of $\vec{x}$ and the model outputs $0.7$, it means the model predicts there is a $70\%$ chance that the tumor is malignant ($y=1$).
* Because the outcome $y$ must be either $0$ or $1$, the combined probabilities must add up to $1$ (or $100\%$).
* Therefore, if there is a $70\%$ chance of $y=1$, there is automatically a $30\%$ chance of $y=0$.

### Mathematical Notation (Optional but common in papers)
* In academic papers or articles, this probability is often written as:
  $$f_{\vec{w},b}(\vec{x}) = P(y=1 | \vec{x} ; \vec{w}, b)$$
* **How to read it:** The probability ($P$) that $y$ equals $1$, given (the vertical line $|$) the input features $\vec{x}$, parameterized by (the semicolon $;$) the parameters $\vec{w}$ and $b$.
