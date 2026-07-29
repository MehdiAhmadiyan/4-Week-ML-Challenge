# 📉 Cost Function in Linear Regression

## 1. Parameters of the Model
* In the linear function $f_{w,b}(x) = wx + b$, the variables $w$ and $b$ are called the **parameters** of the model.
* Parameters are the variables you can adjust during training to improve the model. They are sometimes referred to as coefficients or weights.
* Depending on the values chosen for $w$ and $b$, you get a different function $f(x)$, which generates a different straight line on the graph.
    * **$b$ (y-intercept):** The value where the line crosses the vertical y-axis.
    * **$w$ (slope):** Determines the steepness (slope) of the line.
* The goal of linear regression is to choose values for $w$ and $b$ so that the straight line fits the training data well (i.e., the line passes roughly close to the training examples).

## 2. Defining the Cost Function (Squared Error Cost Function)
To measure how well a line fits the training data, we construct a cost function.
* The cost function takes the model's prediction ($\hat{y}$) and compares it to the actual target ($y$) by taking the difference: $\hat{y} - y$. This difference is called the **error**.
* The cost function measures how far off the prediction is from the target by computing the square of this error.
* It computes this squared error for every training example and sums them all up across the entire training set.
* To ensure the cost doesn't automatically get bigger just because the dataset is larger, we compute the average squared error by dividing the sum by $m$ (the total number of training examples).
* By convention in machine learning, we divide by $2m$ instead of just $m$. The extra division by 2 makes later calculations look neater, but the function works exactly the same either way.

The mathematical expression for the Squared Error Cost Function (denoted as $J$) is:

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

* This specific cost function is called the **squared error cost function**. It is by far the most commonly used cost function for linear regression and generally gives good results for regression problems.

## 3. Visualizing and Minimizing the Cost Function (Intuition)
To build intuition, we can look at a simplified version of the model where the parameter $b$ is set to 0, resulting in the model $f_{w}(x) = wx$.
* In this simplified model, the line always passes through the origin $(0,0)$.
* The cost function $J$ is now only a function of the single parameter $w$: $J(w)$.
* The goal is to find the value for $w$ that minimizes $J(w)$.

### How $f_{w}(x)$ and $J(w)$ are related:

* **The Model Graph** $f_{w}(x)$ : The horizontal axis is the input feature $x$, and the vertical axis is the output target $y$. For a fixed value of $w$, it draws a specific line.
* **The Cost Function Graph** $J(w)$ : The horizontal axis is the parameter $w$, and the vertical axis is the cost $J$. Each point on this graph represents the total error of a specific line drawn on the model graph.

### Step-by-Step Example Calculation:

Assume a training set with three points: $(1,1), (2,2), (3,3)$.

* **Case 1:** $w = 1$
    * The model predicts $f(1)=1$, $f(2)=2$, $f(3)=3$.
    * The error for every point is 0 because the predictions match the actual targets perfectly.
    * The cost $J(1) = 0$. This is the minimum possible cost, meaning $w = 1$ is the best parameter for this specific data.

* **Case 2:** $w = 0.5$
    * The model predicts $f(1)=0.5$, $f(2)=1$, $f(3)=1.5$.
    * There is a visible gap (error) between the predicted points on the line and the actual target points.
    * The squared errors are computed and summed up. For this example, $J(0.5)$ evaluates to approximately $0.58$.

* **Case 3:** $w = 0$
    * The model predicts $f(x)=0$ for all inputs (a flat line on the x-axis).
    * The gap between the predictions and the actual points is very large.
    * For this example, $J(0)$ evaluates to approximately $2.33$.

By plotting the cost $J$ for various values of $w$ (including negative numbers), you trace out a bowl-shaped curve. The ultimate goal of linear regression is to find the parameter(s) at the very bottom of this bowl, which represents the smallest possible value for the cost function $J$.
