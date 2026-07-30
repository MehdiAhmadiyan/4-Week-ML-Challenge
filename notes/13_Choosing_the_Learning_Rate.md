# 🎛️ Choosing the Learning Rate ($\alpha$)

## 1. Identifying Problems with Gradient Descent
A poorly chosen learning rate ($\alpha$) or a bug in your code can cause gradient descent to fail. You can diagnose this by looking at the learning curve (Cost $J$ vs. # of Iterations).

*   **The Zig-Zag Curve (Cost goes up and down):** If the cost sometimes goes up and sometimes goes down, this is a clear sign that gradient descent is not working properly.
    *   **Reason:** The learning rate $\alpha$ is too big. The update step is overshooting the minimum, bouncing from one side of the "bowl" to the higher side on the opposite end, and repeating this process.
    *   **Solution:** Use a smaller learning rate.
*   **The Upward Curve (Cost consistently increases):** If the cost consistently increases after each iteration, this is also likely due to a learning rate that is too large, but it can also be a clear sign of broken code.
    *   **Reason (Code Bug):** For example, if you mistakenly used a plus sign instead of a minus sign in your update rule ($w_1 = w_1 \mathbf{+} \alpha d_1$), the algorithm is mathematically moving the parameter further away from the global minimum. 
    *   **Solution:** Ensure you are using the minus sign for the update.

## 2. A Crucial Debugging Tip
*   With a small enough learning rate $\alpha$, the cost function $J$ **should decrease on every single iteration**.
*   If your gradient descent isn't working, set $\alpha$ to be a very small number as a debugging step.
*   If the cost $J$ still does not decrease consistently even with a tiny $\alpha$, it almost certainly means there is a bug somewhere in your code.
*   *Note:* Using a tiny $\alpha$ is just for debugging. It is not an efficient choice for actual training, because gradient descent will take a massive number of iterations to converge.

## 3. The Strategy for Choosing a Good $\alpha$
Instead of guessing randomly, you should systematically try a range of values for $\alpha$. 

1.  Start with a small value (e.g., $0.001$).
2.  Multiply the previous value by roughly 3 to test the next value. For example, the testing sequence would be:
    $$0.001 \rightarrow 0.003 \rightarrow 0.01 \rightarrow 0.03 \rightarrow 0.1 \rightarrow \dots$$
3.  For each choice of $\alpha$, run gradient descent for just a handful of iterations and plot the cost function.
4.  Continue increasing the value until you find a value that is too large (the cost bounces or increases).
5.  **The Selection:** Once you have found the value that is clearly too large, pick an $\alpha$ that is slightly smaller than that largest reasonable value. This will usually give you a learning rate that decreases the cost rapidly and consistently.
