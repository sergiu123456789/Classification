# 🚗 Random Forest Classification Problem: Predicting New Car Buyers

## 🧩 Problem Statement

A **car company** has released a new model and wants to identify which of their **previous customers** are most likely to purchase it.

To do this, the company uses customer data, specifically:

- **Age**
- **Salary**

Based on this historical data and past purchasing behavior, the goal is to **predict whether a customer will buy the new car** using a **Random Forest Classifier**.

---

## 🎯 Objective

> Train a **Random Forest classification model** to predict whether a customer is likely to purchase the new car model based on their:
> - Age  
> - Estimated Salary

This enables the marketing and sales teams to target only the most promising leads.

---

## 🧪 Dataset Example

| Customer ID | Age | Estimated Salary | Purchased (Target) |
|-------------|-----|------------------|---------------------|
| 101         | 25  | 50,000           | 0                   |
| 102         | 30  | 60,000           | 1                   |
| 103         | 35  | 45,000           | 0                   |
| 104         | 40  | 80,000           | 1                   |
| 105         | 28  | 55,000           | 0                   |

- `Purchased = 1` means the customer bought the car in the past.
- The model will learn patterns and apply them to new customer data.

---

## 🌲 Why Random Forest?

- **Ensemble model** built from multiple decision trees
- Reduces **overfitting**
- Handles both **non-linear relationships** and **interactions** between features
- Offers **feature importance** insights

---

## 🛠️ Workflow

1. **Import libraries and dataset**
2. **Preprocess the data**
   - Split into train/test
   - Feature scaling (if needed)
3. **Train the Random Forest model**
4. **Predict on test/new data**
5. **Evaluate using classification metrics**
6. **Visualize decision boundaries (optional)**

---

## 🧠 Sample Code (Python)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Assume df is your DataFrame
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## ✅ Outcome

- Predict which customers are **most likely to buy** the new car
- Prioritize leads for **targeted marketing and sales**
- Save time and budget by focusing on **high-conversion segments**

---

## 📌 Notes

- You can tune hyperparameters like:
  - `n_estimators` (number of trees)
  - `max_depth`, `min_samples_split`, etc.
- Consider adding more features (e.g., past purchase history, region, credit score) for better performance

---

## 🧠 Summary

By using **Random Forest Classification**, the car company can make **data-driven predictions** about which customers are most likely to buy their new vehicle — improving conversion rates and customer targeting.

