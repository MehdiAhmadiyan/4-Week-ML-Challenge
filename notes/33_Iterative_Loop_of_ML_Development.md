# 🔄 The Iterative Loop of ML Development

## 1. The Iterative Process
* Developing a machine learning system is rarely a straightforward, linear path; it is an iterative loop.
* When you train a model for the very first time, it will almost never work as well as you want it to.
* The development process involves cycling through three main steps:
  1. **Choose Architecture:** Decide on the overall architecture, which includes choosing the machine learning model, the data to use, and hyperparameters.
  2. **Train Model:** Implement and train the algorithm based on your initial choices.
  3. **Diagnostics:** Look at diagnostics such as bias, variance, and error analysis to understand why the model is falling short.
* Based on the insights gained from diagnostics, you update your architecture (e.g., make the neural network bigger, change $\lambda$, add features) and go around the loop again until you reach the desired performance.

## 2. Practical Example: Spam Classifier
* To build a system that recognizes spam versus non-spam emails, we use a supervised learning algorithm for text classification.
* The output label $y$ is $1$ (spam) or $0$ (non-spam).
* **Feature Extraction:** You can list the top 10,000 most common words in the English language to define features $x_1$ through $x_{10,000}$.
* For a given email, you create a feature vector $\vec{x}$ where each element is $1$ if the specific word appears in the email, and $0$ if it does not (or you can count the frequency of the word).
* Using these features, you can train a logistic regression model or a neural network.

## 3. Brainstorming Improvements
If the initial spam classifier does not work well, there are many potential ideas to reduce the error:
* **Collect more data:** Create "Honeypot" projects, which involve setting up fake email addresses deliberately given to spammers to collect massive amounts of spam data.
* **Email routing features:** Develop sophisticated features based on the email header, which tracks the path of servers the email traveled through.
* **Email body features:** Make the text processing smarter (e.g., treating "discount" and "discounting" as the exact same word).
* **Detecting misspellings:** Spammers often deliberately misspell words (e.g., "w4tches", "med1cine", "m0rtgage") to bypass filters; designing algorithms to detect these can improve accuracy.

## 4. Deciding What to Try Next
* Having many ideas is great, but choosing the most promising path forward can speed up your project by 10 times.
* For example, if diagnostics show your algorithm suffers from **High Bias**, spending months on a Honeypot project to collect more data will be completely useless. However, if it suffers from **High Variance**, collecting more data is highly effective.
* In addition to Bias and Variance, another critical diagnostic tool for making these decisions is called **Error Analysis**.
