# 📉 Gradient Descent for Logistic Regression

## 1. The Gradient Descent Algorithm
* To fit the parameters of a logistic regression model, we try to find the specific values of $\vec{w}$ and $b$ that minimize the cost function $J(\vec{w},b)$.
* To achieve this minimization, we apply the Gradient Descent algorithm.
* The algorithm repeatedly applies the following updates until it converges:

$$w_j = w_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})x_j^{(i)} \right]$$

$$b = b - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}) \right]$$

* Just like in linear regression, it is absolutely critical to use **simultaneous updates**. 
* You must compute the right-hand side expressions for all parameters first, and then simultaneously overwrite all the old values of $w_j$ and $b$ with the new ones.

## 2. The Big Surprise: It Looks Identical!
* You might notice something weird: the derivative formulas and the update equations for logistic regression look exactly identical to the ones used for linear regression.
* However, linear regression is not secretly the same as logistic regression. 
* Even though the algorithms written down look the same, they are fundamentally different because the mathematical definition of the prediction function $f_{\vec{w},b}(\vec{x})$ has changed.
* **In Linear Regression:** $f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$.
* **In Logistic Regression:** $f_{\vec{w},b}(\vec{x}) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$ (The Sigmoid Function).

## 3. Reusing Past Concepts
Many of the powerful concepts you learned for linear regression apply directly to logistic regression in the exact same way.

*   **Monitoring Gradient Descent:** You can monitor the learning curve (plotting the cost $J$ against the number of iterations) to ensure gradient descent is converging properly and to check your learning rate $\alpha$.
*   **Vectorization:** Instead of updating the parameters $w_j$ one by one using a slow loop, you can use vectorization to implement the updates simultaneously, making the algorithm run much faster.
*   **Feature Scaling:** Scaling all your features so they take on similar ranges of values (e.g., between -1 and +1) will speed up gradient descent and help it converge significantly faster.
*   **Practical Implementation:** While it is important to know how to write this from scratch, many machine learning practitioners regularly use popular libraries like **Scikit-Learn** as part of their day-to-day jobs to train logistic regression models.
