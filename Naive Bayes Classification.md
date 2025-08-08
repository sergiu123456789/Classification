# 🚗 Naive Bayes Classification Problem: Predicting Car Purchases

## 🧩 Problem Statement

A **car company** has just released a new model and wants to **predict which of its previous customers** are likely to **purchase the new car**.

The company has historical data that includes:
- **Customer Age**
- **Customer Salary**
- **Whether or not they purchased a car in the past**

Using this data, we want to train a **Naive Bayes classifier** that can predict whether a given customer will buy the **new car model**.

---

## 🎯 Objective

> Use the **Naive Bayes algorithm** to build a binary classification model that can classify customers as:
>
> - `1` → Will buy the new car model  
> - `0` → Will not buy the new car model

---

## 🧪 Dataset Example

| Age | Salary (USD) | Purchased (Target) |
|-----|--------------|--------------------|
| 22  | 35,000       | 0                  |
| 35  | 60,000       | 1                  |
| 47  | 85,000       | 1                  |
| 52  | 52,000       | 0                  |
| 28  | 48,000       | 0                  |

---

## 🤖 Approach

### Step 1: Import Libraries and Load Data
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

### Step 2: Prepare the Dataset
```python
# Load data
data = pd.read_csv("car_customer_data.csv")

# Features and target
X = data[['Age', 'Salary']]
y = data['Purchased']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### Step 3: Train Naive Bayes Model
```python
model = GaussianNB()
model.fit(X_train, y_train)
```

### Step 4: Make Predictions
```python
y_pred = model.predict(X_test)
```

### Step 5: Evaluate Model
```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

## 🧠 Why Naive Bayes?

- **Fast and simple** to implement
- Works well with **binary classification**
- Makes the **naive assumption** that features (age and salary) are **independent**, which simplifies the computation

---

## ✅ Outcome

- Identify likely buyers for marketing campaigns
- Focus resources on high-potential leads
- Use **age and salary** to make data-driven decisions

---

## 🔍 Notes

- You can improve accuracy by scaling features or adding more (e.g., location, car ownership history)
- GaussianNB is used here, assuming normally distributed features
- Works best when features are **not heavily correlated**

---

## 📌 Summary

> Build a **Naive Bayes classifier** to predict which previous customers are likely to buy the newly released car based on their **age** and **salary**.

This enables the company to **target the right audience** and **optimize marketing spend** using a lightweight but effective ML model.
