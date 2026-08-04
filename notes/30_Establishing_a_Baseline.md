# 📏 Establishing a Baseline Level of Performance

## 1. The Problem with "Zero Error" Expectation
* In previous videos, we assumed that if the training error ($J_{train}$) is high, the model automatically has a High Bias problem.
* However, in many real-world tasks (like speech recognition), the data itself might be extremely noisy or ambiguous (e.g., audio with heavy background noise). 
* Even a human expert cannot perfectly transcribe this noisy data, making it impossible and unrealistic to expect a machine learning algorithm to achieve 0% error.

## 2. What is a Baseline Level of Performance?
* To accurately judge if an error rate is actually "high", you need to establish a **baseline level of performance**. 
* **Definition:** The baseline is the level of error you can reasonably hope your learning algorithm will eventually achieve.
* **How to establish it?**
    1.  **Human Level Performance:** Often the best benchmark, especially for unstructured data like audio, images, or text, because humans are naturally very good at processing these.
    2.  **Competing Algorithms:** Looking at the performance of an older system or a competitor's algorithm.
    3.  **Educated Guess:** Based on prior experience in the specific domain.

## 3. How to Diagnose Using the Baseline
Instead of just looking at the absolute value of $J_{train}$, you must compare the gaps between three numbers: the **Baseline**, **$J_{train}$**, and **$J_{cv}$**.

*   **Gap 1 (Bias Indicator):** The difference between the Baseline and $J_{train}$. If this gap is large, you have a **High Bias** problem.
*   **Gap 2 (Variance Indicator):** The difference between $J_{train}$ and $J_{cv}$. If this gap is large, you have a **High Variance** problem.

## 4. Concrete Examples (Speech Recognition)

Assume the baseline (Human Level Performance) is **10.6% error**.

### Example A: High Variance
*   **Baseline:** 10.6%
*   **$J_{train}$:** 10.8%
*   **$J_{cv}$:** 14.8%
*   *Diagnosis:* The gap between Baseline and $J_{train}$ is only 0.2% (The algorithm is doing great on the training set, almost matching humans!). However, the gap between $J_{train}$ and $J_{cv}$ is a massive 4.0%. This indicates a **High Variance** problem.

### Example B: High Bias
*   **Baseline:** 10.6%
*   **$J_{train}$:** 15.0%
*   **$J_{cv}$:** 15.5%
*   *Diagnosis:* The gap between Baseline and $J_{train}$ is 4.4% (The model is performing significantly worse than humans on the training set). The gap between $J_{train}$ and $J_{cv}$ is only 0.5%. This indicates a **High Bias** problem.

### Example C: High Bias AND High Variance
*   **Baseline:** 10.6%
*   **$J_{train}$:** 15.0%
*   **$J_{cv}$:** 19.7%
*   *Diagnosis:* Both gaps are huge. The model struggles to fit the training data (Gap of 4.4%), and it struggles even more to generalize to the cross-validation data (Gap of 4.7%). This indicates the model has **both High Bias and High Variance**.
