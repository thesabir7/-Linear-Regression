# 📘 Day 3: Linear Regression – From Scratch vs Scikit-learn

As part of my **10-Day Machine Learning Challenge**, I implemented and compared Linear Regression models—both **from scratch** and using **scikit-learn**—on the **Boston Housing dataset**.

---

## 📌 What is Linear Regression?

**Linear Regression** is one of the simplest and most widely used algorithms in supervised machine learning.  
It assumes a **linear relationship** between the input variable(s) (features) and the output (target).

### 🔸 Equation of a Line:

\[
y = mx + c
\]

In **Linear Regression**, this becomes:

\[
\hat{y} = w_1 x + w_0
\]

- \( \hat{y} \): predicted value  
- \( x \): input feature(s)  
- \( w_1 \): weight (slope)  
- \( w_0 \): bias (intercept)

The model **learns** the best values for \( w_1 \) and \( w_0 \) by minimizing the **cost function** (usually **Mean Squared Error**).

---

## 🧠 Learning Outcomes

- Implemented **gradient descent** to optimize weights
- Compared with `LinearRegression` from **scikit-learn**
- Understood how RMSE and R² score help evaluate performance
- Visualized model predictions vs actual target values

---

## 🔍 Dataset Used

- **Boston Housing Dataset**
  - Features: We used only the `RM` feature (average number of rooms per dwelling)
  - Target: `MEDV` (Median house value in $1000s)
  - Available directly via `sklearn.datasets.load_boston()`
