# ⚙️ The Anomaly Detection Algorithm

## 1. Handling Multiple Features (Density Estimation)
*   In real-world applications, a single training example ($x$) usually has $n$ different features (e.g., $x_1$ = heat, $x_2$ = vibration). 
*   This means each example is actually an $n$-dimensional vector.
*   To build an anomaly detection system, we must estimate the probability of the entire feature vector, denoted as $p(x)$.
*   We do this by calculating the individual probability of each feature and then **multiplying them all together**. 

*(Note: Mathematically, this assumes the features are statistically independent, but in practice, the algorithm works perfectly fine even if they are not)*.

**The Intuition:**
If there is a $\frac{1}{10}$ chance an engine runs unusually hot, and a $\frac{1}{20}$ chance it vibrates unusually hard, the chance of an engine doing *both* at the same time is their product: $\frac{1}{10} \times \frac{1}{20} = \frac{1}{200}$. 

We write this mathematically using the product symbol ($\prod$):

$$
p(x) = \prod_{j=1}^{n} p(x_j ; \mu_j, \sigma_j^2)
$$

## 2. The Complete Algorithm Steps

**Step 1: Choose Features**
*   Select $n$ features ($x_i$) that you believe might effectively indicate when an example is anomalous.

**Step 2: Fit the Parameters**
*   For every single feature $j$ (from 1 to $n$), you must calculate its mean ($\mu_j$) and variance ($\sigma_j^2$) based on the training dataset.

To calculate the mean for feature $j$:
$$
\mu_j = \frac{1}{m} \sum_{i=1}^{m} x_j^{(i)}
$$

To calculate the variance for feature $j$:
$$
\sigma_j^2 = \frac{1}{m} \sum_{i=1}^{m} (x_j^{(i)} - \mu_j)^2
$$

*(Vectorized implementation allows you to compute the mean vector $\mu$ for all features simultaneously by averaging the vectors)*.

**Step 3: Compute $p(x)$ for a New Example**
*   When a brand new test example ($x$) arrives, you plug its features into the Gaussian probability formula and multiply them all together.

$$
p(x) = \prod_{j=1}^{n} \frac{1}{\sqrt{2\pi}\sigma_j} \exp\left( - \frac{(x_j - \mu_j)^2}{2\sigma_j^2} \right)
$$

**Step 4: Flag Anomalies**
*   Finally, compare the calculated $p(x)$ against a very small threshold number called $\epsilon$ (epsilon).
*   If $p(x) < \epsilon \rightarrow$ **Flag as an Anomaly**.
*   If $p(x) \ge \epsilon \rightarrow$ **Normal (OK)**.

## 3. Concrete Example (Visualized in 3D)
Let's look at the aircraft engine dataset with two features:
*   $x_1$ (Heat): Mean $\mu_1 = 5$, Standard Deviation $\sigma_1 = 2$.
*   $x_2$ (Vibration): Mean $\mu_2 = 3$, Standard Deviation $\sigma_2 = 1$.

When you multiply the two Gaussian curves $p(x_1)$ and $p(x_2)$ together, you create a 3D surface plot. The peak of this 3D mountain represents the highest probability (normal engines). The flat areas far away from the peak represent very low probability (anomalies).

Assume we set our threshold $\epsilon = 0.02$.
*   **Test Engine 1 (Near the center):** $p(x_{test}^{(1)}) = 0.0426$. Since $0.0426 \ge 0.02$, the algorithm declares it **OK**.
*   **Test Engine 2 (Far out on the graph):** $p(x_{test}^{(2)}) = 0.0021$. Since $0.0021 < 0.02$, the algorithm correctly flags it as an **Anomaly**.
