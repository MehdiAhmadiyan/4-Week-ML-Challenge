# 📏 Evaluating a Model

## 1. Why Do We Need Systematic Evaluation?
* When training a model, fitting the training data perfectly (like using a 4th-order polynomial) often results in a wiggly curve that fails to generalize to new, unseen examples.
* If you only have one feature (e.g., house size), you can easily plot the function and visually see if it is overfitting.
* However, if you have many features (e.g., size, bedrooms, floors, age), plotting a multi-dimensional function becomes impossible. 
* Therefore, we need a systematic mathematical way to evaluate how well a model is doing.

## 2. The Train / Test Split
To systematically evaluate a model, we split the entire dataset into two separate subsets:
*   **Training Set (e.g., 70% of data):** The model is trained and its parameters ($w, b$) are fitted using *only* this portion of the data. The number of training examples is denoted as $m_{train}$.
*   **Test Set (e.g., 30% of data):** This data is kept hidden during training. It is used strictly to test the model's performance on new examples. The number of test examples is denoted as $m_{test}$.

## 3. Evaluation for Linear Regression
When using linear regression (with squared error cost), the procedure is as follows:

1. **Fit the Parameters:** Minimize the regularized cost function $J(\vec{w},b)$ using the training set to find $\vec{w}$ and $b$.

2. **Compute Test Error ($J_{test}$):** 

$$
J_{test}(\vec{w},b) = \frac{1}{2m_{test}} \sum_{i=1}^{m_{test}} (f_{\vec{w},b}(\vec{x}_{test}^{(i)}) - y_{test}^{(i)})^2
$$

*(Note: The test error formula does **not** include the regularization term)*.

3. **Compute Training Error ($J_{train}$):**

$$
J_{train}(\vec{w},b) = \frac{1}{2m_{train}} \sum_{i=1}^{m_{train}} (f_{\vec{w},b}(\vec{x}_{train}^{(i)}) - y_{train}^{(i)})^2
$$

*(Note: The training error formula also does **not** include the regularization term)*.

*   **Diagnosing Overfitting:** If a model overfits, $J_{train}$ will be very low (close to zero), but there will be a large gap when making predictions on the test set, making $J_{test}$ very high. Seeing a high $J_{test}$ means the model is failing to generalize.

## 4. Evaluation for Classification (Logistic Regression)
When classifying data (e.g., 0 or 1), the procedure is conceptually similar but uses different metrics.

*   **Fit the Parameters:** Minimize the regularized logistic cost function $J(\vec{w},b)$ on the training set.
*   **Method A (Logistic Loss):** You can calculate $J_{test}$ and $J_{train}$ using the average logistic loss function across the test set and training set, respectively (again, without the regularization term).
*   **Method B (Fraction of Misclassification):** A more common and intuitive way to evaluate classification is to measure the fraction of misclassified examples.
    *   Make a prediction ($\hat{y} = 1$ if $f \ge 0.5$, else $\hat{y} = 0$).
    *   Count how many times $\hat{y} \ne y$ (the prediction does not equal the actual ground truth).
    *   $J_{test}$ becomes the exact fraction (percentage) of the test set that was misclassified.
    *   $J_{train}$ becomes the fraction of the training set that was misclassified.
