# 📈 Developing and Evaluating an Anomaly Detection System

## 1. The Need for "Real-Number Evaluation"
*   When developing a machine learning system, you constantly need to make decisions (e.g., choosing new features, or increasing/decreasing the threshold $\epsilon$).
*   Making these decisions is much easier if you have a way to evaluate your algorithm and get a single number that tells you if the algorithm just got better or worse. This is called **Real-Number Evaluation**.

## 2. Using Labeled Data in an Unsupervised Algorithm
Even though anomaly detection is an unsupervised learning algorithm, to properly evaluate it, we need a small amount of labeled data. 
*   $y = 0$: Normal (Non-anomalous) examples.
*   $y = 1$: Flawed (Anomalous) examples.

We use a large dataset of normal examples to *train* the model, but we use a small mix of normal and anomalous examples to *evaluate* and *tune* it.

## 3. Data Splitting: Train, CV, and Test Sets
Let's use the aircraft engine example. Suppose over the years you have manufactured 10,000 normal engines, but you have found 20 flawed (anomalous) engines. 

Here is how you optimally split this data:

*   **Training Set:** 6,000 Normal engines ($y=0$). 
    *   *(You fit the Gaussian distribution model $p(x)$ exclusively on this set)*.
*   **Cross-Validation (CV) Set:** 2,000 Normal engines ($y=0$) + 10 Anomalous engines ($y=1$).
    *   *(You use this set to tune the parameter $\epsilon$ and decide which features to include)*.
*   **Test Set:** 2,000 Normal engines ($y=0$) + 10 Anomalous engines ($y=1$).
    *   *(You use this set to perform the final evaluation of your system)*.

**Alternative Split (If Anomalies are Extremely Rare):**
If you only have, for example, 2 flawed engines in total, you cannot create a separate Test set. You would use 6,000 normal engines for Training, and put the remaining 4,000 normal + 2 anomalous engines entirely into the CV set. *(Warning: This has a higher risk of overfitting your $\epsilon$ parameter to the CV set)*.

## 4. Algorithm Evaluation Metrics
Once the model is trained, you evaluate it on the Cross-Validation or Test set using this prediction rule:
*   Predict $y = 1$ (Anomaly) if $p(x) < \epsilon$.
*   Predict $y = 0$ (Normal) if $p(x) \ge \epsilon$.

**Why Classification Accuracy Fails:**
In the CV set, you have 2,000 negative examples and only 10 positive examples. This is a highly **skewed dataset**. An algorithm that blindly predicts "$y=0$" for everything would achieve 99.5% accuracy, but it would completely fail its actual job (finding the anomalies).

**The Right Metrics to Use:**
Instead of accuracy, you must use the evaluation metrics designed specifically for skewed datasets (which we covered in Supervised Learning):
*   True Positives, False Positives, False Negatives, True Negatives.
*   **Precision and Recall**.
*   **F1-Score**.

**The Goal:** You test multiple different values for $\epsilon$ on your CV set, calculate the F1-Score for each, and pick the $\epsilon$ that yields the highest F1-Score.
