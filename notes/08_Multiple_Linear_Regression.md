# 🏡 Multiple Linear Regression

## 1. Moving from One to Multiple Features
* In the original linear regression model, we had a single feature $x$ (e.g., the size of the house) to predict $y$ (the price).
* Now, we can use multiple features such as size, number of bedrooms, number of floors, and age of the home to get much more information for predicting the price.

## 2. New Notation
To handle multiple variables, we need to introduce some new mathematical notation:
* $x_j$: Represents the value of the $j^{th}$ feature.
* $n$: Denotes the total number of features in the dataset.
* $\vec{x}^{(i)}$: Represents a list of numbers (a vector) that includes all the features of the $i^{th}$ training example. It is technically called a row vector.
* $x_j^{(i)}$: Refers to the value of a specific feature $j$ in the $i^{th}$ training example.
* Drawing a little arrow on top of variables (like $\vec{x}$ or $\vec{w}$) is an optional signifier used to visually emphasize that the variable represents a list of numbers (a vector) rather than a single number.

## 3. The Multiple Linear Regression Model
* With $n$ features, the model is expanded and defined as: $f_{\vec{w},b}(\vec{x}) = w_1x_1 + w_2x_2 + w_3x_3 + ... + w_nx_n + b$.

**Parameter Interpretation (Housing Example):**
* $b$: The base value of the prediction if all features are zero (e.g., a base starting price of $80,000 for a house with zero size, bedrooms, floors, and age).
* $w_1, w_2, ...$: These parameters represent the impact or weight of each individual feature on the prediction. For example, the price might increase by a certain amount for each additional bedroom ($w_2$), or decrease for each additional year of age ($w_4$ being a negative number).

## 4. Vector Notation and the Dot Product
* To write the model expression in a simpler, more compact form with fewer characters, we define parameters and features as vectors.
* $\vec{w} = [w_1, w_2, w_3, ..., w_n]$ is a vector containing all the weight parameters.
* $b$ remains a single number and is not a vector.
* $\vec{x} = [x_1, x_2, x_3, ..., x_n]$ is a vector containing all the features.
* Using linear algebra, the model can be rewritten using a **dot product**: $f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$.
* A dot product is computed by multiplying corresponding pairs of numbers from the two vectors ($w_1x_1$, $w_2x_2$, etc.) and then summing all of these products together. 
* This dot product calculation gives the exact same mathematical expression as the expanded model form.

## 5. Terminology Clarification
* A linear regression model with multiple input features is officially called **Multiple Linear Regression**.
* This is in contrast to univariate regression, which relies on just one feature.
* You should not call this "multivariate regression," because that term actually refers to a completely different concept in statistics that will not be used here.
* To implement this model efficiently in code, we use a trick called **vectorization**, which simplifies the implementation of this and many other learning algorithms.
