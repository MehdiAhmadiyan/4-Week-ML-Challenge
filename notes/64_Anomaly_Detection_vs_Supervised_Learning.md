# ⚖️ Anomaly Detection vs. Supervised Learning

If we have labeled data indicating normal events ($y=0$) and flawed events ($y=1$), deciding whether to use Anomaly Detection or Supervised Learning can be a subtle choice. Here is the framework to help you decide.

## 1. The Amount of Data (The Numbers Game)
The most immediate difference is the volume of available data.

*   **Anomaly Detection:** You use this when you have a **very small** number of positive examples ($y=1$). Having only 0 to 20 positive examples is very common. You rely on a large number of negative examples ($y=0$) to build your $p(x)$ model.
*   **Supervised Learning:** You use this when you have a **large number** of both positive and negative examples in your dataset.

## 2. The Core Difference: The "Nature" of the Problem
Even if you have 20 positive examples, the fundamental way these algorithms look at the data is different.

*   **Anomaly Detection (Expecting the Unknown):** You use this when there are many different "types" of anomalies. It is hard for an algorithm to learn what an anomaly looks like from just 20 examples because future anomalies might look **nothing like** any of the anomalies you have seen so far. 
    *   *How it thinks:* "I will perfectly learn what *normal* looks like. If anything looks weird or different from normal, I will flag it."
*   **Supervised Learning (Expecting the Familiar):** You use this when you assume that future positive examples will be **similar to** the positive examples already in your training set. You have enough positive examples for the algorithm to get a clear sense of what they look like.
    *   *How it thinks:* "I have seen a thousand pictures of this specific defect. I will look for this exact same defect in future examples."

## 3. Real-World Examples and Comparisons

### Fraud and Security
*   **Anomaly Detection (Financial Fraud / Data Center Hacking):** Hackers and criminals constantly invent brand-new, unique ways to steal money or attack servers. Because the attacks keep changing, you want to flag *any* highly unusual behavior.
*   **Supervised Learning (Email Spam):** Spam emails generally use the same patterns over and over (trying to sell similar things or steal passwords). Because future spam looks like past spam, supervised learning works perfectly.

### Manufacturing
*   **Anomaly Detection (Aircraft Engines):** There are countless ways a complex engine could fail, and a brand-new failure method might happen tomorrow. You want to find *new, previously unseen defects*.
*   **Supervised Learning (Smartphone Cases):** You know your manufacturing machine frequently causes a specific defect (like scratching the phone case). You have thousands of examples of scratched phones, so you train a model to specifically find *known, previously seen defects*.

### General Categories
*   **Supervised Learning** is strictly used for things like predicting the weather (sunny, rainy, snowy) or disease classification, where the output labels are consistent and repeated over time.
