# ⚖️ Feature Scaling

## 1. Why Do We Need Feature Scaling?
When features in a dataset take on very different ranges of values (e.g., $x_1$ ranges from 300 to 2000, while $x_2$ ranges from 0 to 5), it causes severe problems for Gradient Descent.
*   **The Parameter Imbalance:** A feature with a large range of values (like size) will tend to have a relatively small parameter value (e.g., $w_1 = 0.1$). Conversely, a feature with a small range of values (like bedrooms) will tend to have a relatively large parameter value (e.g., $w_2 = 50$).
*   **The Problem with Contours:** Because of this imbalance, a very small change in $w_1$ has a massive impact on the cost $J$, while a large change in $w_2$ has a much smaller impact.
*   This causes the contour plot of the cost function to look like a very tall, skinny ellipse (oval).
*   **The Consequence:** If you run Gradient Descent on these tall, skinny contours, the algorithm will bounce back and forth erratically for a very long time before finally finding its way to the global minimum.

## 2. What is Feature Scaling?
Feature scaling is a transformation technique that rescales different features so they all take on comparable ranges of values.
*   When features are scaled, the contour plot of the cost function becomes much more circular (like circles instead of skinny ovals).
*   With circular contours, Gradient Descent can find a much more direct, straight path to the global minimum, causing the algorithm to run significantly faster.

## 3. How to Implement Feature Scaling (Three Methods)
There are three common ways to scale your features:

### A. Dividing by the Maximum
You simply take every original value of a feature and divide it by the maximum value found in that feature's column.
*   **Formula:** $x_{scaled} = \frac{x}{x_{max}}$
*   *Example:* If $x_1$ ranges from 300 to 2000, dividing every value by 2000 will scale $x_1$ to range from 0.15 to 1.

### B. Mean Normalization
This method centers the data around zero so it has both positive and negative values (usually between -1 and +1).
*   **Formula:** $x_{scaled} = \frac{x - \mu}{max - min}$
*   Here, $\mu$ (mu) is the average (mean) of the feature. You subtract the mean from the value, and divide by the difference between the maximum and minimum values.

### C. Z-score Normalization (Standardization)
This method uses the standard deviation ($\sigma$) and the mean ($\mu$) to scale the data.
*   **Formula:** $x_{scaled} = \frac{x - \mu}{\sigma}$
*   You subtract the mean from the value and divide it by the standard deviation. If you do this, the scaled values will typically range from around -3 to +3.

## 4. Rules of Thumb for Scaling
*   The general goal is to get all features to range from roughly **-1 to +1**.
*   These boundaries are a little bit loose. Ranges like -3 to +3, or -0.3 to +0.3 are completely fine.
*   If a feature naturally ranges from 0 to 3, or -2 to +0.5, you can leave it alone; there is no harm in skipping scaling for those.
*   **When you MUST scale:**
    *   If a feature is too large (e.g., -100 to +100).
    *   If a feature is too small (e.g., -0.001 to +0.001).
    *   If a feature has a large offset from zero (e.g., 98.6 to 105).
*   **The Golden Rule:** There is almost never any harm in carrying out feature scaling. When in doubt, just do it.
