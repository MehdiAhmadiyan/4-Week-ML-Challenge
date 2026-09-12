# 🚨 Anomaly Detection: Finding Unusual Events

## 1. What is Anomaly Detection?
*   Anomaly detection is another highly commercially successful **unsupervised learning** algorithm.
*   Instead of grouping data into clusters, its job is to look at an unlabeled dataset of "normal" events and learn to raise a red flag if it encounters an unusual or anomalous event.

## 2. The Aircraft Engine Example
To understand this visually, let's look at an aircraft engine manufacturing plant.
*   After an engine is built, we measure its features. For example:
    *   $x_1$ = heat generated
    *   $x_2$ = vibration intensity
*   We collect data from $m$ normal, perfectly fine engines: $\{x^{(1)}, x^{(2)}, \dots, x^{(m)}\}$.
*   When we plot these engines on a graph, they form a general group (a dense area) in the center.
*   **The Test:** When a brand new engine ($x_{\text{test}}$) rolls off the assembly line, we plot its measurements on the exact same graph.
    *   If $x_{\text{test}}$ lands right in the middle of our normal data points, it is probably OK.
    *   If $x_{\text{test}}$ lands far away from the normal points, it is flagged as an **anomaly**. This engine must be carefully inspected manually before it is ever installed on an airplane.

## 3. How it Works: Density Estimation
To mathematically achieve this, the algorithm uses a technique called **Density Estimation**.
*   **Step 1:** The algorithm builds a model for the probability of $x$, written as $p(x)$. It learns which combinations of features have a high probability (meaning they are common/normal) and which have a low probability (meaning they are rare/unusual).
*   **Step 2:** We define a threshold using a very small number called **epsilon** ($\epsilon$).
*   **Step 3:** When given a new test example ($x_{\text{test}}$), the algorithm computes $p(x_{\text{test}})$:
    *   If $p(x_{\text{test}}) < \epsilon \rightarrow$ **Flag as an anomaly** (The probability of this happening naturally is too low).
    *   If $p(x_{\text{test}}) \ge \epsilon \rightarrow$ **Normal** (It looks OK).

## 4. Real-World Applications
This technique is used silently everywhere in the tech industry:
*   **Fraud Detection:** Websites track user features (login frequency, pages visited, typing speed). If a user's behavior suddenly drops to a $p(x) < \epsilon$, the system doesn't immediately ban them, but flags the account for additional security checks (like sending an SMS verification code).
*   **Manufacturing:** Checking smartphones, circuit boards, and motors for strange defects before shipping them to customers.
*   **Data Centers:** Monitoring computers by checking memory use, disk access per second, CPU load, and network traffic ratios. If a computer acts weirdly, it might indicate a hardware failure or a cyberattack (hacking).
