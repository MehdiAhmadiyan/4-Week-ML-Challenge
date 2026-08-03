# 🚧 The Decision Boundary

## 1. Making a Prediction (The Threshold)
* The logistic regression model outputs a probability between 0 and 1.
* To make a final decision and predict whether the class $y$ is 0 or 1, we typically set a threshold at 0.5.
* If $f_{\vec{w},b}(\vec{x}) \ge 0.5$, the algorithm predicts $\hat{y} = 1$.
* If $f_{\vec{w},b}(\vec{x}) < 0.5$, the algorithm predicts $\hat{y} = 0$.

## 2. The Math Behind the Decision
We know that the model is defined as $f_{\vec{w},b}(\vec{x}) = g(z)$, where $g$ is the Sigmoid function and $z = \vec{w} \cdot \vec{x} + b$.
* When is $g(z) \ge 0.5$? 
* Looking at the Sigmoid curve, the output is greater than or equal to 0.5 whenever $z \ge 0$.
* Therefore, the model predicts $\hat{y} = 1$ whenever $\vec{w} \cdot \vec{x} + b \ge 0$.
* Conversely, the model predicts $\hat{y} = 0$ whenever $\vec{w} \cdot \vec{x} + b < 0$.

## 3. What is the Decision Boundary?
* The **Decision Boundary** is the exact line or mathematical boundary where you are completely neutral about the prediction (where the output is exactly 0.5).
* This occurs exactly when $\vec{w} \cdot \vec{x} + b = 0$.
* The decision boundary separates the entire graph into two distinct regions: one region where the model predicts 1, and another region where it predicts 0.

## 4. Linear Decision Boundaries
* If you only use standard features (like $x_1$ and $x_2$) without any higher-order polynomials, the decision boundary for logistic regression will **always be a straight line**.
* **Example:** Suppose we have two features ($x_1, x_2$) and parameters $w_1 = 1$, $w_2 = 1$, and $b = -3$.
* The decision boundary is calculated as: $x_1 + x_2 - 3 = 0$.
* This simplifies to $x_1 + x_2 = 3$, which draws a straight diagonal line across the graph.

## 5. Non-linear Decision Boundaries
* Just like in linear regression, you can use **feature engineering** to include polynomial terms (like $x_1^2$, $x_2^2$, $x_1x_2$, etc.) into logistic regression.
* Using polynomial features allows the logistic regression model to learn complex, non-linear decision boundaries.
* **Example (A Circle):** Suppose we define $z = w_1x_1^2 + w_2x_2^2 + b$, and we set $w_1 = 1$, $w_2 = 1$, and $b = -1$.
    * The decision boundary occurs when $x_1^2 + x_2^2 - 1 = 0$.
    * This simplifies to $x_1^2 + x_2^2 = 1$, which is the mathematical equation for a circle.
    * The model will predict $\hat{y} = 0$ inside the circle and $\hat{y} = 1$ outside the circle.
* By adding even higher-order polynomials (e.g., cubes or interactions like $x_1x_2$), you can create incredibly complex decision boundaries, such as ellipses or completely irregular, free-form shapes.
