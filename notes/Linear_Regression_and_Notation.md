# 📈 Linear Regression and Supervised Learning Process

## 1. Introduction to Linear Regression
* Linear regression involves fitting a straight line to your data.
* It is a type of supervised learning because the model is trained on data that already contains the "right answers".
* For example, to predict the price of a house, you feed the model examples of houses with both their size and their actual sold price. 

## 2. Regression vs. Classification (A Quick Review)
* **Regression Model:** Predicts numbers as the output, meaning there are infinitely many possible numbers the model could output (e.g., predicting a house price like $220,000 or -33.2).
* **Classification Model:** Predicts discrete categories, meaning there is a small, finite set of possible outputs (e.g., predicting if a picture is a cat or a dog, or if a patient has a specific disease).

## 3. Standard Machine Learning Notation
To describe data and mathematical concepts, Machine Learning uses standard notation:
* **Training Set:** The dataset that is used to train the model.
* **$x$:** Denotes the input variable, which is also called a feature or input feature (e.g., size of the house).
* **$y$:** Denotes the output variable or target variable that you are trying to predict (e.g., price of the house).
* **$m$:** Refers to the total number of training examples in the dataset.
* **$(x, y)$:** Represents a single training example.
* **$(x^{(i)}, y^{(i)})$:** Refers to a specific, $i^{th}$ training example. The superscript $i$ is just an index representing the row in the data table, not an exponentiation.

## 4. How the Supervised Learning Process Works
* The supervised learning algorithm takes the training set (input features and output targets) as input and produces a function, historically called a hypothesis, but generally denoted as $f$.
* This function $f$ is called the model.
* The job of $f$ is to take a new input $x$ and output a prediction, denoted as $\hat{y}$ (y-hat).
* While $y$ refers to the actual true value (target) in the training set, $\hat{y}$ represents the model's estimated value or prediction.

## 5. Mathematical Representation of Linear Regression
* When using a straight line, the math formula for the function is written as $f_{w,b}(x) = wx + b$.
* It can also be written more simply as $f(x) = wx + b$.
* The letters $w$ and $b$ represent numbers.
* The values chosen for the numbers $w$ and $b$ determine the prediction $\hat{y}$ based on the input $x$.
* Linear regression with a single input variable is officially called **univariate linear regression**.
* The prefix "uni" means one in Latin, and "variate" means variable.
* A linear function (straight line) is relatively simple and easy to work with, acting as a foundation before moving on to more complex non-linear models.
* In order to make linear regression work, the most important next step is to construct a **cost function**.
