# 🔍 Choosing What Features to Use (Anomaly Detection)

## 1. Why Feature Selection is Critical
*   In supervised learning, if you include irrelevant features, it is usually fine. The algorithm uses the target labels ($y$) to figure out which features to ignore and which to scale.
*   In anomaly detection (unsupervised learning), there are no labels to guide the algorithm. It cannot automatically figure out what to ignore. Therefore, carefully choosing and engineering your features is much more critical for success.

## 2. Tip 1: Transforming Non-Gaussian Features
The anomaly detection algorithm models $p(x)$ using a Gaussian (bell-shaped) distribution. If your raw features do not look Gaussian, the model will not fit the data well.

*   **Check the distribution:** Always plot a histogram of your feature.
*   **Apply transformations:** If the histogram is heavily skewed (not symmetric), you can mathematically transform the data to make it look more Gaussian.

Common transformations include:

$$x \leftarrow \log(x)$$
$$x \leftarrow \log(x + c)$$
$$x \leftarrow \sqrt{x}$$
$$x \leftarrow x^{1/3}$$

*(Important: Whatever transformation you apply to your training set, you must apply the exact same transformation to your cross-validation and test sets!)*

## 3. Tip 2: Error Analysis (Adding New Features)
*   **The Problem:** The most common issue is that $p(x)$ turns out to be large for *both* normal examples and anomalous examples. The algorithm fails to spot the anomaly because its current features look normal.
*   **The Solution:** Look specifically at the anomalous example in your Cross-Validation set that the algorithm missed. Ask yourself: *"What makes this specific example anomalous?"* 
*   **Example (Fraud Detection):** A fraudulent user might make a normal number of transactions (so feature $x_1$ looks perfectly fine). However, you notice they are typing insanely fast. By creating a brand new feature ($x_2$ = typing speed), this user will suddenly stand out completely on the graph, allowing the algorithm to easily flag them.

## 4. Tip 3: Combining Existing Features
Sometimes, individual features look normal, but their *relationship* is highly unusual. You can create powerful new features by combining (e.g., dividing or multiplying) existing ones.

*   **Example (Data Center Monitoring):** 
    *   $x_3$ = CPU load
    *   $x_4$ = Network traffic
*   Normally, if a server is doing a lot of work (high CPU load), it is also streaming a lot of data (high network traffic). 
*   If a machine gets stuck in an infinite loop (a localized anomaly), it might have a very high CPU load but almost zero network traffic. Looking at CPU or Network alone might not trigger a red flag, because high CPU loads happen naturally.
*   **The Fix:** Create a new feature that captures the ratio between them.

$$x_5 = \frac{\text{CPU load}}{\text{Network traffic}}$$

By doing this, $x_5$ will produce an unusually huge number for the stuck machine, causing $p(x)$ to drop drastically and successfully flagging the anomaly.
