# 🚗 SVM Classification Problem: Predicting Car Purchase Likelihood

## 🧩 Problem Statement

A **car company** has released a new model and wants to identify **which previous customers are most likely to buy it**.

The company has historical data on customer demographics, including:

- **Age**
- **Salary**
- Whether or not they purchased **previous car models**

Using this data, the goal is to build a **classification model** to predict if a customer will buy the **new model**.

---

## 🎯 Objective

> Use **Support Vector Machine (SVM)** with a **linear kernel** to predict whether a customer will purchase the new car model based on their **age** and **salary**.

---

## 📊 Dataset Example

| Customer ID | Age | Salary (USD) | Purchased Previous Model (Yes/No) |
|-------------|-----|---------------|----------------------------------|
| 001         | 22  | 25,000        | No                               |
| 002         | 45  | 85,000        | Yes                              |
| 003         | 31  | 40,000        | No                               |
| 004         | 35  | 95,000        | Yes                              |
| ...         | ... | ...           | ...                              |

The target variable is **binary**:
- `1` = Will likely buy
- `0` = Will not buy

---

## 🧠 Why Use SVM (Linear Kernel)?

- Ideal for **binary classification**
- Performs well with **linearly separable data**
- Robust to high-dimensional data
- Effective for small to medium-sized datasets

---

## 🧪 Model Development Steps

1. **Import libraries and dataset**
2. **Preprocess data** (handle missing values, encode target variable)
3. **Feature scaling** (essential for SVM)
4. **Split data into training and test sets**
5. **Train an SVM classifier with a linear kernel**
6. **Predict test results**
7. **Evaluate the model** using metrics like:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
8. **Visualize decision boundary**

---

## 🧾 Sample Code (SVM with Linear Kernel)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Sample features and labels
X = dataset[['Age', 'Salary']].values
y = dataset['Purchased'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train SVM classifier (linear kernel)
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## ✅ Outcome

- Identify which customers are **most likely to buy**
- Focus marketing campaigns on **high-potential buyers**
- Reduce acquisition costs and improve conversion rates

---

## 📌 Notes

- Can later experiment with non-linear kernels (e.g., RBF) for more complex patterns
- Consider including additional features: **location**, **marital status**, **previous purchases**, etc.
- Use **cross-validation** to improve generalization

---

## 📦 Business Value

By applying **SVM classification**, the car company can:
- Anticipate buyer behavior
- Personalize marketing
- Boost ROI on new model launches
