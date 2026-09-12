# 🔔 The Gaussian (Normal) Distribution

## 1. What is the Gaussian Distribution?
*   The Gaussian distribution, also known as the Normal distribution, is a mathematical way to describe how data is spread out. 
*   When plotted on a graph, it forms a symmetrical **bell-shaped curve**.
*   It is defined by two key parameters: the mean, denoted by $\mu$ (which sets the center of the curve), and the variance, denoted by $\sigma^2$ (which determines how wide or narrow the curve is). The square root of the variance is $\sigma$, known as the standard deviation.

## 2. The Probability Formula and Curve Shape
The formula to calculate the probability of a specific number $x$ occurring, written as $p(x)$, is:

$$p(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{\frac{-(x-\mu)^2}{2\sigma^2}}$$

Changing the parameters drastically alters the shape of this bell curve:
*   **Decreasing $\sigma$ (e.g., from 1 to 0.5):** The curve becomes much skinnier and taller, meaning the data is tightly packed around the center.
*   **Increasing $\sigma$ (e.g., from 1 to 2):** The curve becomes much wider and shorter, meaning the data is spread out over a larger range.
*   **Changing $\mu$:** The entire curve shifts left or right along the axis without changing its width or height.

## 3. Parameter Estimation (Calculating from Data)
In machine learning, you are given a dataset with $m$ examples and you need to estimate the best $\mu$ and $\sigma^2$ to fit your data. 

To estimate the center (mean) of your data, you calculate the average of all examples:

$$\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}$$

To estimate the spread (variance) of your data, you calculate the average of the squared differences between each example and the mean:

$$\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu)^2$$

*(Note: Some statisticians divide by $m-1$ instead of $m$, but in machine learning, dividing by $m$ is standard practice and makes very little practical difference)*.

## 4. Applying it to Anomaly Detection
*   Once you have estimated $\mu$ and $\sigma^2$ from your dataset of normal events, you have a working probability model.
*   If a new test example lands near the center ($\mu$), the formula outputs a high $p(x)$, meaning the event is normal and common.
*   If a new test example lands far out in the "tails" of the curve, the formula outputs a very low $p(x)$, triggering a red flag because the event is highly unusual (an anomaly).
