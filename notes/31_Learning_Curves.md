# 📈 Learning Curves

## 1. What are Learning Curves?
Learning curves are visual tools used to understand how a learning algorithm's performance changes as a function of its experience (i.e., the number of training examples, $m_{train}$). 

When we plot the errors against $m_{train}$, we observe two general behaviors:
*   **Cross-Validation Error ($J_{cv}$):** As the training set size gets bigger, the algorithm learns a better model, so $J_{cv}$ generally **goes down**.
*   **Training Error ($J_{train}$):** Surprisingly, as the training set gets bigger, the training error actually **increases**. 
    *   *Why?* If you only have 1, 2, or 3 examples, it is incredibly easy for a model (like a quadratic curve) to fit them perfectly, resulting in zero error. But as you add more and more data points, it becomes much harder to fit every single point perfectly, causing the average training error to rise.

## 2. Learning Curves for High Bias (Underfitting)
If you plot the learning curves for a model with high bias (e.g., fitting a straight line to complex data), you will see a specific pattern:

*   **The Shape:** Both $J_{train}$ and $J_{cv}$ will rise/fall and then quickly **plateau (flatten out)**. 
*   **The Gap to Baseline:** There will be a significant gap between the flattened curves and the desired baseline (e.g., human-level performance).
*   **Why it flattens:** Because the model is too simple (like a straight line), no matter how much data you feed it, the model's shape won't change much. It simply cannot capture the complexity of the data.
*   **The Golden Rule for High Bias:** If a learning algorithm suffers from high bias, getting more training data will **NOT** (by itself) help much. The curve is flat; adding more data to the right of the plot will just extend the flat line without lowering the error.

## 3. Learning Curves for High Variance (Overfitting)
If you plot the learning curves for a model with high variance (e.g., fitting a very high-order polynomial with small $\lambda$), the pattern looks completely different:

*   **The Shape:** $J_{train}$ stays very low (sometimes even below human-level performance because it memorizes the training data). $J_{cv}$ remains high.
*   **The Gap between Errors:** There is a **huge gap** between $J_{cv}$ and $J_{train}$.
*   **The Golden Rule for High Variance:** If a learning algorithm suffers from high variance, getting more training data is **likely to help**. If you extrapolate the curves to the right (adding more data), you can see $J_{cv}$ continuing to trend downwards, eventually approaching $J_{train}$ and the baseline.

## 4. Practical Implementation Note
*   To actually plot these curves, you would need to train your model multiple times on increasingly larger subsets of your data (e.g., train on 100 examples, then 200, then 300) and plot the errors.
*   Because this process is computationally expensive, it is not done very frequently in everyday practice. 
*   However, keeping the **mental visual picture** of these curves in your head is an incredibly powerful diagnostic tool to decide whether collecting more data is worth your time and money.
