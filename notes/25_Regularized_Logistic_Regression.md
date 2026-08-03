# 🛡️ Regularized Logistic Regression

## 1. Overfitting in Logistic Regression
* Logistic regression can be highly prone to overfitting if you fit the model with very high-order polynomial features.
* Passing a complex polynomial into the sigmoid function can result in a highly contorted, overly complex decision boundary that overfits the training set.
* When training logistic regression with a lot of features, the risk of overfitting increases significantly.

## 2. The Regularized Cost Function
* To modify the logistic regression cost function to use regularization, we simply add the standard regularization term to the end of the equation.
* The new regularized cost function is:

$$J(\vec{w},b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(f_{\vec{w},b}(\vec{x}^{(i)})) + (1 - y^{(i)}) \log(1 - f_{\vec{w},b}(\vec{x}^{(i)}))] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

* Minimizing this new cost function penalizes the parameters $w_1$ through $w_n$, effectively preventing them from becoming too large.
* Even when fitting a high-order polynomial, this penalty ensures you get a smooth, reasonable decision boundary that generalizes well to new, unseen examples.

## 3. Implementing Gradient Descent
* To minimize this regularized cost function, we continue to use the Gradient Descent algorithm with simultaneous updates.
* The update rules are exactly the same as the ones used for regularized linear regression:

$$w_j = w_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})x_j^{(i)} + \frac{\lambda}{m} w_j \right]$$

$$b = b - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}) \right]$$

* **What Changed?** The derivative with respect to $w_j$ simply gets the additional term $\frac{\lambda}{m} w_j$ added at the end.
* **Parameter $b$:** Just like in linear regression, we only regularize the parameters $w_j$ and do not regularize the parameter $b$. This is why the update rule for $b$ remains completely unchanged.

## 4. The Golden Rule
* While the gradient descent update equations look mathematically identical to regularized linear regression, the algorithms are fundamentally different.
* The critical difference lies in the definition of the prediction function. In logistic regression, $f_{\vec{w},b}(\vec{x})$ is the logistic (sigmoid) function applied to $z$, not a simple linear function.
