# 📉 Cost Function for Logistic Regression

## 1. The Problem with Squared Error
* In linear regression, the squared error cost function naturally creates a smooth, bowl-shaped (convex) surface.
* If you try to use that exact same squared error cost function for logistic regression (by plugging in the Sigmoid function), the resulting cost surface becomes **non-convex**.
* A non-convex function looks "wiggly" and has many local minima. 
* If you run Gradient Descent on a non-convex surface, the algorithm can easily get stuck in a local minimum and fail to find the best possible parameters.

## 2. The Logistic Loss Function (Single Example)
To fix the non-convexity problem, we define a new way to measure error for a *single* training example, which is called the **Loss Function** ($L$).

*   **When the true target $y=1$:**
    $$L(f_{\vec{w},b}(\vec{x}), y) = -\log(f_{\vec{w},b}(\vec{x}))$$
    *   *Intuition:* If the model predicts a value very close to 1, the loss approaches 0 (which is good). If the model predicts a value close to 0 (completely wrong), the loss shoots up to infinity, heavily penalizing the model.
*   **When the true target $y=0$:**
    $$L(f_{\vec{w},b}(\vec{x}), y) = -\log(1 - f_{\vec{w},b}(\vec{x}))$$
    *   *Intuition:* If the model predicts a value very close to 0, the loss approaches 0. If the model is highly confident but wrong (predicts close to 1), the loss again goes to infinity.

## 3. The Simplified Loss Function
Writing the loss function in two separate cases (if $y=1$, if $y=0$) makes coding it inefficient. Because $y$ can strictly only take the value of 0 or 1, we can mathematically compress both cases into a single, elegant equation:

$$L(f_{\vec{w},b}(\vec{x}), y) = -y \log(f_{\vec{w},b}(\vec{x})) - (1 - y) \log(1 - f_{\vec{w},b}(\vec{x}))$$

*   **Why this works:**
    *   If $y=1$, the second part of the equation is multiplied by $(1 - 1) = 0$ and completely disappears, leaving only the $-\log(f)$ part.
    *   If $y=0$, the first part of the equation is multiplied by 0 and disappears, leaving only the $-\log(1 - f)$ part.

## 4. The Final Cost Function $J(\vec{w},b)$
The overall Cost Function ($J$) is simply the average of the loss function across all $m$ training examples in the entire dataset. 

By plugging our simplified loss function into the average formula, we get the definitive cost function used by almost everyone to train logistic regression models:

$$J(\vec{w},b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(f_{\vec{w},b}(\vec{x}^{(i)})) + (1 - y^{(i)}) \log(1 - f_{\vec{w},b}(\vec{x}^{(i)}))]$$

*   **Key Property:** This specific cost function is mathematically guaranteed to be **convex** (a perfect bowl shape). Because of this, Gradient Descent will always reliably converge to the global minimum.
*   **Origin:** This function isn't arbitrary; it is derived from a statistical principle called **Maximum Likelihood Estimation**, which is a proven method for efficiently finding parameters for different models.
