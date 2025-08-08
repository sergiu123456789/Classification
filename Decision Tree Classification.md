# 🚗 Decision Tree Classification Problem: Predicting Car Purchase

## 🧩 Problem Statement

A **car company** has released a **new model** and wants to identify **which of its previous customers are likely to buy it**.

To optimize their **marketing and sales strategy**, they aim to predict customer interest based on:

- **Age**
- **Salary**

Using historical data, the company wants to build a **Decision Tree Classifier** that can predict whether a customer is likely to **buy** or **not buy** the new car model.

---

## 🎯 Objective

> Train a **Decision Tree Classification** model using previous customer data to predict whether a customer will buy the new car model, based on their age and salary.

---

## 🧪 Dataset Example

| Age | Salary (USD) | Purchased (Yes/No) |
|-----|--------------|--------------------|
| 25  | 50,000       | No                 |
| 30  | 60,000       | Yes                |
| 35  | 45,000       | No                 |
| 40  | 80,000       | Yes                |
| 22  | 30,000       | No                 |
| 50  | 90,000       | Yes                |

The target variable is **Purchased**, a binary classification (Yes = 1, No = 0).

---

## 🤖 How It Works

The Decision Tree algorithm splits the dataset into smaller subsets based on conditions like:

- If age < 35
- If salary > $65,000

Each decision creates a branch in the tree, eventually leading to a prediction at a leaf node: **Buy** or **Not Buy**.

---

## 🧠 Python Code Example (Simplified)

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
X = [[25, 50000], [30, 60000], [35, 45000], [40, 80000], [22, 30000], [50, 90000]]
y = [0, 1, 0, 1, 0, 1]  # 1 = Buy, 0 = Not Buy

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## ✅ Outcome

- 🎯 Predict which customers are most likely to buy the new car model
- 💰 Save marketing budget by targeting only high-likelihood buyers
- 📊 Support decision-making with explainable rules from the decision tree

---

## 📂 Notes

- Try pruning or setting `max_depth` to avoid overfitting
- Add more features (e.g., previous purchases, location, car preferences) to improve accuracy

---

## 📌 Business Value

By using **Decision Tree Classification**, the car company can:

- 📈 Increase conversion rates by targeting likely buyers
- 🧠 Make informed marketing decisions
- 🔍 Understand the "why" behind predictions via the tree's structure

