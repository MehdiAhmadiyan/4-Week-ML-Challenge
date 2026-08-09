# 🔍 Error Analysis

## 1. What is Error Analysis?
* While diagnosing Bias and Variance is the most important tool for improving a learning algorithm, **Error Analysis** is arguably the second most important.
* Error analysis is the process of manually examining a set of examples that your algorithm has misclassified, in order to gain insights into where exactly it is going wrong.
* This manual process often provides the inspiration needed to decide which specific changes or improvements will be most fruitful to try next.

## 2. The Process of Error Analysis
*   **Step 1:** Run your trained model on the Cross-Validation set. Let's say you have $m_{cv} = 500$ examples, and the algorithm misclassifies $100$ of them.
*   **Step 2:** Manually look through those 100 misclassified examples one by one.
*   **Step 3:** Try to group them into common themes, traits, or categories. Keep a manual count of how many errors fall into each category.

### A Concrete Example (Spam Classifier)
Suppose you manually analyze your 100 misclassified spam emails and group them into these categories:
*   Pharma spam (selling medicine): **21** errors
*   Stealing passwords (phishing): **18** errors
*   Unusual email routing: **7** errors
*   Spam message hidden in an embedded image: **5** errors
*   Deliberate misspellings (e.g., w4tches): **3** errors

*(Note: These categories are not mutually exclusive; one email could be both pharma spam and have deliberate misspellings)*.

## 3. How to Use the Results
The numbers you gather from Error Analysis directly dictate your project priorities:

*   **Prioritize High-Impact Fixes:** The analysis clearly shows that Pharma and Phishing emails are huge problems. This tells you that collecting *specific* data just for Pharma/Phishing, or developing new features specifically to detect drug names or suspicious URLs, will significantly improve your model.
*   **Avoid Low-Impact Traps:** You might have thought that building sophisticated algorithms to detect misspellings was a great idea. However, the analysis shows it only accounts for 3 out of 100 errors. Even if you build a perfect misspelling detector, the net impact on your overall error rate will be tiny. Error analysis saves you from wasting months on unfruitful work.

## 4. Practical Considerations and Limitations
*   **Sampling Large Datasets:** If your cross-validation set has 5,000 examples and your algorithm misclassifies 1,000 of them, you likely don't have time to look at all 1,000. In practice, you randomly sample a subset of around 100 examples to analyze manually, which is usually enough to give you reliable statistics.
*   **Limitation:** Error analysis is much easier for tasks that humans are naturally good at (like reading emails, looking at images, or hearing audio). It is much harder to do for tasks humans are bad at (e.g., predicting exactly which ad a user will click on).
