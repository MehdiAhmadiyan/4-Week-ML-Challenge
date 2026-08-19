# 🛠️ Feature Engineering

## 1. What is Feature Engineering?
* The choice of features can have a massive impact on the performance of your learning algorithm.
* For many practical applications, choosing or engineering the right features is a critical step to making the algorithm work well.
* **Definition:** Feature engineering is the process of using your knowledge or intuition about a problem to design new features. 
* This is usually done by transforming or combining the original features of the problem to make it easier for the algorithm to make accurate predictions.

## 2. A Practical Example (Housing Prices)
* Suppose you are predicting house prices and have two original features: $x_1$ (the width or frontage of the lot) and $x_2$ (the depth of the lot).
* A standard multiple linear regression model would look like this: $f_{\vec{w},b}(\vec{x}) = w_1x_1 + w_2x_2 + b$.
* **Applying Intuition:** You might realize that the total area of the land is a much stronger predictor of price than the width and depth measured separately.
* **Engineering the Feature:** You can define a brand new feature $x_3$ by multiplying the two original features: $x_3 = x_1 \times x_2$.
* The new model becomes: $f_{\vec{w},b}(\vec{x}) = w_1x_1 + w_2x_2 + w_3x_3 + b$.
* By doing this, the model can now choose parameters $w_1$, $w_2$, and $w_3$ based on what the data shows is most important, often resulting in a much better model.
