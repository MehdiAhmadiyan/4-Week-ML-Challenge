# 📈 Checking Gradient Descent Convergence

## 1. The Learning Curve
* To check if gradient descent is working properly and finding parameters close to the global minimum, you can plot the cost function $J$.
* This specific plot is called a **learning curve**.
* The horizontal axis represents the number of iterations of gradient descent you have run so far.
* This is completely different from previous graphs where the horizontal axis represented a specific parameter like $w$ or $b$.
* The vertical axis represents the value of the cost function $J$, which is calculated on the training set after each simultaneous update of the parameters.
* For example, a point on the graph at 100 iterations shows the exact value of the cost $J$ using the learned parameters $w$ and $b$ obtained after exactly 100 updates.

## 2. Recognizing Proper Behavior & Issues
* If the gradient descent algorithm is working properly, the cost $J$ should decrease after every single iteration.
* If $J$ ever increases after one iteration, it means there is a problem: either there is a bug in your code, or the learning rate $\alpha$ is chosen poorly (which usually means $\alpha$ is too large).
* As the iterations progress, the curve will eventually level off and flatten out; this indicates that gradient descent has more or less converged.

## 3. The Number of Iterations Needed
* The number of iterations that gradient descent takes to converge can vary a lot between different applications.
* In one application, it may converge after just 30 iterations, while in a different application, it could take 1,000 or even 100,000 iterations.
* It turns out to be very difficult to tell in advance how many iterations the algorithm will need to converge, which is why creating a learning curve graph is so useful.

## 4. Automatic Convergence Test
* Another way to decide when your model is done training is by using an automatic convergence test.
* You can define a variable $\epsilon$ (epsilon) to represent a very small number, such as 0.001 ($10^{-3}$).
* If the cost $J$ decreases by an amount less than this $\epsilon$ on a single iteration, you are likely on the flattened part of the curve and can declare that the algorithm has converged.
* **Practical Advice:** Choosing the right threshold for $\epsilon$ is pretty difficult in practice. It is generally better to look at the visual learning curve graph rather than relying on automatic convergence tests, because the visual plot can also give you advanced warning if gradient descent is not working correctly.
