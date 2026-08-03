# 📊 Introduction to Classification

## 1. What is Classification?
* Unlike linear regression, which predicts any number from an infinite range of numbers, classification is used when the output variable $y$ can take on only one of a small handful of possible values.
* **Binary Classification:** This is a specific type of classification problem where there are exactly two possible outputs or categories.
* Common examples of binary classification include predicting whether an email is spam, whether an online financial transaction is fraudulent, or whether a tumor is malignant.

## 2. Terminology and Notation
In binary classification, the two categories are usually represented by specific terms:
* The categories are often designated as "no" or "yes", or "false" or "true".
* Following computer science conventions, these are most commonly represented by the numbers $0$ and $1$.
* **$0$ (Negative Class):** Represents the "false" or "no" category.
* **$1$ (Positive Class):** Represents the "true" or "yes" category.
* *Important Note:* The terms "negative" and "positive" do not mean "bad" versus "good" or "evil" versus "good". They merely convey the absence ($0$) or presence ($1$) of the specific property you are looking for (e.g., the presence of the spam property or the presence of malignancy).

## 3. Why Linear Regression Fails for Classification
You might wonder if you can just use the linear regression algorithm ($f_{w,b}(x) = wx + b$) to solve a classification problem. Here is why that is a bad idea:

*   **The Threshold Approach:** Since linear regression predicts continuous numbers, you could try setting a threshold at $0.5$. 
    *   If the model outputs a value $< 0.5$, you predict $\hat{y} = 0$.
    *   If the model outputs a value $\ge 0.5$, you predict $\hat{y} = 1$.
*   The vertical point where this threshold intersects the best-fit line acts as a dividing line (called the **decision boundary**).
*   **The Outlier Problem:** If you add a single new training example that is far to the right (e.g., a massive malignant tumor), it acts as an outlier and pulls the best-fit linear regression line down.
*   **The Consequence:** This shift causes the decision boundary to also shift over to the right. As a result, data points that were previously classified correctly can suddenly be misclassified, making linear regression unreliable for classification tasks.

## 4. Enter Logistic Regression
* To overcome the severe limitations of linear regression in categorizing data, we use a different algorithm called **Logistic Regression**.
* In logistic regression, the output value of the algorithm is mathematically constrained to always fall between $0$ and $1$.
* *A Note on the Name:* Do not let the name confuse you; despite having the word "regression" in its name for historical reasons, Logistic Regression is actually one of the most widely used algorithms for solving binary classification problems today.
