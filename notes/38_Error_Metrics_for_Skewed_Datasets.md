# 📉 Error Metrics for Skewed Datasets

## 1. The Problem with "Accuracy" in Skewed Data
* A dataset is "skewed" when the ratio of positive ($y=1$) to negative ($y=0$) examples is extreme (e.g., very far from 50-50).
* **Example:** Detecting a rare disease where only $0.5\%$ of patients actually have it ($y=1$).
* **The Illusion of Accuracy:** If you write a "dumb" piece of software that ignores the data and simply always prints `y = 0`, it will achieve a $99.5\%$ accuracy (or $0.5\%$ error). 
* Even though the accuracy is extremely high, the algorithm is completely useless because it never successfully detects a single case of the disease. Thus, simple accuracy is a terrible metric for skewed datasets.

## 2. The Confusion Matrix
To properly evaluate an algorithm on skewed data, we designate the rare class (the disease we want to detect) as $y=1$ and construct a $2 \times 2$ "Confusion Matrix":

*   **True Positive (TP):** We predicted 1, and the actual class was 1. (We correctly diagnosed the disease).
*   **True Negative (TN):** We predicted 0, and the actual class was 0. (We correctly said the patient is healthy).
*   **False Positive (FP):** We predicted 1, but the actual class was 0. (We falsely alarmed a healthy patient).
*   **False Negative (FN):** We predicted 0, but the actual class was 1. (We missed a sick patient).

## 3. Precision
**Precision** answers the question: *Of all the patients where we predicted $y=1$, what fraction actually has the rare disease?*

$$
\text{Precision} = \frac{\text{True Positives}}{\text{Total Predicted Positive}}
$$
$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

*   If an algorithm has high precision, it means that when it raises the alarm, it is usually right.

## 4. Recall
**Recall** answers the question: *Of all the patients that actually have the rare disease, what fraction did we correctly detect?*

$$
\text{Recall} = \frac{\text{True Positives}}{\text{Total Actual Positive}}
$$
$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

*   If an algorithm just prints `y = 0` all the time, its True Positives will be $0$. Therefore, its Recall will drop to precisely $0\%$, instantly exposing it as a useless algorithm despite its high accuracy.

## 5. The Goal
When dealing with a rare class, a learning algorithm is only considered genuinely useful if both **Precision** and **Recall** are reasonably high.
