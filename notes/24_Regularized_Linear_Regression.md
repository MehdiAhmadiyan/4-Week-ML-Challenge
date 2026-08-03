# ⚙️ Gradient Descent for Regularized Linear Regression

## 1. The New Update Rules
To minimize the regularized cost function $J(\vec{w},b)$, we continue to use the Gradient Descent algorithm, but with a slight modification to the derivative of $w_j$. 

*   The update rules now look like this (repeated until convergence):

$$w_j = w_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})x_j^{(i)} + \frac{\lambda}{m} w_j \right]$$

$$b = b - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}) \right]$$

*   **What changed?** The only difference is the addition of the term $\frac{\lambda}{m} w_j$ at the end of the $w_j$ update equation.
*   **What about $b$?** Because we do not regularize the parameter $b$, its derivative and its update rule remain exactly the same as in unregularized linear regression.
*   **Important Reminder:** You must still carry out **simultaneous updates** for all parameters $w_1 \dots w_n$ and $b$.

## 2. Deep Intuition: How Regularization Shrinks $w_j$
To truly understand how this math magically prevents overfitting, we can rearrange the terms in the $w_j$ update rule. 

If we multiply $\alpha$ into the brackets and factor out $w_j$, the equation can be rewritten as:
$$w_j = w_j \left( 1 - \alpha \frac{\lambda}{m} \right) - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})x_j^{(i)}$$

*   Notice that the second part of this equation is just the **usual update** from standard linear regression.
*   The magic happens in the first part: $w_j \left( 1 - \alpha \frac{\lambda}{m} \right)$.
*   **A Numerical Example:** Suppose the learning rate $\alpha = 0.01$, the regularization parameter $\lambda = 1$, and the number of training examples $m = 50$.
    *   $\alpha \frac{\lambda}{m} = 0.01 \times \frac{1}{50} = 0.0002$.
    *   So, $\left( 1 - \alpha \frac{\lambda}{m} \right) = 1 - 0.0002 = 0.9998$.
*   **The Conclusion:** On every single iteration of gradient descent, before applying the usual update, the algorithm multiplies $w_j$ by a number slightly less than $1$ (e.g., $0.9998$). This has the exact mathematical effect of continually shrinking the value of $w_j$ just a little bit, which is exactly how regularization works to prevent overfitting.

## 3. The Calculus Behind the Derivative (Optional)
If you are curious about where the new term came from, it is a straightforward application of calculus.
*   The regularization term in the cost function is $\frac{\lambda}{2m} \sum w_j^2$.
*   When calculating the partial derivative with respect to $w_j$, the exponent $2$ comes down and cancels out the $2$ in the denominator ($2m$).
*   This clean cancellation is exactly why the term simplifies perfectly to $\frac{\lambda}{m} w_j$.
