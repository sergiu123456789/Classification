# 🚗 K-Nearest Neighbors (KNN) Problem: Predicting Car Purchase Likelihood

## 🧩 Problem Statement

A **car company** is planning to launch a **new model** and wants to predict which of its **existing customers** are most likely to purchase it.

To make this prediction, the company will use **historical data** about customers’:
- **Age**
- **Salary**
- **Purchase Decision** (whether they bought previous models)

The goal is to use this information to train a **K-Nearest Neighbors (KNN)** classifier and predict whether a new or existing customer is likely to purchase the newly released car model.

---

## 🎯 Objective

> Use **K-Nearest Neighbors (KNN)** to predict whether a customer will purchase the new car model, based on their **age** and **salary**.

---

## 📚 Dataset Example

| Age | Salary (USD) | Purchased (Yes=1 / No=0) |
|-----|--------------|---------------------------|
| 22  | 25,000       | 0                         |
| 35  | 65,000       | 1                         |
| 47  | 85,000       | 1                         |
| 52  | 45,000       | 0                         |
| 29  | 55,000       | 1                         |
| ... | ...          | ...                       |

This data will be used to train a **KNN classifier** that can generalize to new inputs.

---

## 🤖 KNN Model Training

Steps:
1. Import required libraries and dataset
2. Preprocess data (e.g. feature scaling)
3. Split the dataset into training and test sets
4. Train the **KNN model**
5. Predict on new/unseen customers
6. Evaluate using metrics like **accuracy**, **confusion matrix**, **F1-score**

---

## 🧪 Sample Prediction

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Example dataset
X = dataset[['Age', 'Salary']].values
y = dataset['Purchased'].values

# Scale features
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict on new customer
new_customer = sc.transform([[40, 60000]])
prediction = knn.predict(new_customer)
print("Will Purchase" if prediction[0] == 1 else "Will Not Purchase")
```

---

## ✅ Outcome

- 🔍 Identify high-probability customers to **target marketing efforts**
- 📈 Optimize sales strategy for the **new car launch**
- 🧠 Apply a simple yet effective ML algorithm (KNN) to make customer-level predictions

---

## 📂 Notes

- Model accuracy depends on **value of `k`**, which can be tuned using cross-validation
- **Feature scaling is mandatory** for KNN to work properly
- Additional features (like location, previous purchases, brand loyalty) could improve prediction

---

## 🧠 Summary

Using **K-Nearest Neighbors**, the car company can:
- Predict new model purchase behavior
- Leverage customer data (age + salary)
- Focus marketing efforts on the most promising leads

📌 This enables **data-driven decision-making** in product launches and customer retention.

