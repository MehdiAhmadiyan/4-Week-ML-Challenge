# 🛠️ Addressing Overfitting

## 1. Option 1: Collect More Training Data
* The number one tool you can use against overfitting is to get more training data.
* With a larger training set, the learning algorithm will learn to fit a function that is less "wiggly" and smoother.
* If you have enough training examples, you can continue to use high-order polynomials or functions with many features, and the model will still generalize well.
* **Limitation:** Getting more data isn't always an option (e.g., if only a limited number of houses have been sold in a specific location).

## 2. Option 2: Feature Selection (Use Fewer Features)
* If you have a large number of features (e.g., 100 features) but not enough training data, your algorithm is highly likely to overfit.
* You can address this by selecting and using only a smaller subset of the most relevant features (e.g., just size, bedrooms, and age).
* You can use your intuition to choose this subset, or use algorithms designed to automatically select the best features (which are taught later in the specialization).
* Setting a parameter to exactly 0 is mathematically equivalent to eliminating that feature outright.
* **Disadvantage:** By eliminating features, you are throwing away some information that might actually be useful for making predictions.

## 3. Option 3: Regularization
* Regularization is a very useful technique to more gently reduce the impacts of features without doing something as harsh as eliminating them outright.
* It works by encouraging the learning algorithm to shrink the values of the parameters (making the $w_j$ values very small) without demanding they be set exactly to 0.
* By keeping the parameter values small, it prevents the features from having an overly large effect on the prediction.
* Even if you fit a high-order polynomial, using small parameter values results in a curve that fits the training data much better and avoids wild fluctuations.
* **Convention for $b$:** By convention, we typically only regularize the weights ($w_1$ through $w_n$) and do not encourage the parameter $b$ to become smaller. In practice, it makes very little difference whether you regularize $b$ or not.
