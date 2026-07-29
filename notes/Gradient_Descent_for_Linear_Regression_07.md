# 🚀 Gradient Descent for Linear Regression

## 1. The Complete Update Equations
* By combining the linear regression model, the squared error cost function, and the gradient descent algorithm, we get the complete update rules.
* You repeatedly carry out the following updates until convergence:

$$w = w - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}$$

$$b = b - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})$$

* In these formulas, $f_{w,b}(x^{(i)})$ represents the linear regression model, which is equal to $wx^{(i)} + b$.
* It is critical to remember that the updates for $w$ and $b$ must be computed and applied simultaneously on each step.

## 2. The Calculus Derivation (Optional Math)
* The derivative formulas used in the update rules are derived using the rules of calculus.
* The original squared error cost function is defined as $J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$.
* When calculating the partial derivative with respect to $w$, the exponent $2$ comes down and cancels out the $2$ in the $\frac{1}{2m}$ denominator.
* Due to the chain rule in calculus, taking the derivative with respect to $w$ leaves an $x^{(i)}$ multiplied at the end of the expression.
* Calculating the derivative with respect to $b$ follows a very similar process, but leaves no $x^{(i)}$ at the end.
* This mathematical cancellation is exactly why the cost function was originally defined with a division by $2$ (the $1.5$ or $1/2m$ factor); it simply makes the final partial derivative formulas neater.

## 3. The Convex Cost Function
* A common issue with gradient descent is that it can sometimes lead to a local minimum instead of the global minimum.
* The global minimum is the absolute lowest possible value for the cost function $J$ across all possible points.
* However, when using the squared error cost function specifically for linear regression, the function will never have multiple local minima.
* It has a single global minimum because it takes the shape of a bowl.
* The formal technical term for this bowl-shaped function is a **convex function**.
* When gradient descent is run on a convex function, it will always converge to the global minimum, provided that the learning rate is chosen appropriately.

## 4. Batch Gradient Descent
* As the algorithm runs, the cost decreases at each update step, and the parameters $w$ and $b$ follow a trajectory straight toward the global minimum.
* Reaching this global minimum means the algorithm has successfully found the straight line that best fits the training data.
* This specific implementation of the algorithm is called **Batch Gradient Descent**.
* The term "batch" refers to the fact that on every single step of gradient descent, the algorithm looks at all of the training examples in the dataset.
* This exhaustive look at the data is mathematically represented by computing the sum from $i=1$ to $m$ at each update.
* While there are other versions of gradient descent that look at smaller subsets of data, standard linear regression uses the entire batch of training examples at each step.
