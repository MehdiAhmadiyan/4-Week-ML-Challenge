# 🛠️ Deciding What to Try Next (Debugging a Learning Algorithm)

## 1. The Core Philosophy of Debugging
* When your learning algorithm makes unacceptably large errors, you should not just randomly guess what to do next.
* The absolute first step is always to look at the Training Error ($J_{train}$) and Cross-Validation Error ($J_{cv}$) to diagnose whether your algorithm suffers from **High Bias** or **High Variance**.
* Once you have your diagnosis, you can systematically apply the correct fix.

## 2. Fixes for High Variance (Overfitting)
If your model has High Variance (it fits the training data well, but fails to generalize to the cross-validation set), you need to either feed it more data or **simplify the model** to reduce its flexibility.

*   **Get more training examples:** This helps the algorithm learn a more generalized pattern rather than memorizing a small dataset.
*   **Try smaller sets of features:** By eliminating irrelevant or redundant features, you restrict the model's flexibility to fit overly complex, wiggly curves.
*   **Try increasing $\lambda$:** Increasing the regularization parameter forces the algorithm to shrink the parameters ($w_j$), resulting in a smoother, simpler function that is less prone to overfitting.

## 3. Fixes for High Bias (Underfitting)
If your model has High Bias (it fails to even fit the training data properly), you need to give the model **more power and flexibility** to fit complex functions.

*   **Try getting additional features:** If you are trying to predict house prices using *only* size, the model lacks enough information. Adding features like bedrooms, floors, and age gives the model the power to make better predictions.
*   **Try adding polynomial features ($x_1^2, x_2^2, x_1x_2$, etc.):** If a straight line isn't enough, adding polynomial features allows the model to bend and curve to fit the training data better.
*   **Try decreasing $\lambda$:** Decreasing the regularization parameter reduces the penalty on the parameters ($w_j$), allowing the algorithm to pay more attention to minimizing the training error rather than keeping the parameters small.

## 4. What NOT to do!
*   **Never reduce the training set size to fix High Bias!** While throwing away data might technically make it easier to fit the remaining few examples (lowering $J_{train}$), it will almost certainly worsen your cross-validation error and the overall performance of the model.

> **A Note from the Field:** Bias and Variance is a concept that "takes a short time to learn, but takes a lifetime to master". Diagnosing your models systematically is a skill that will profoundly improve your efficiency as an AI engineer.
