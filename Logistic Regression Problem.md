# 🚗 Logistic Regression Problem: Predicting Car Purchase Intent

## 🧩 Problem Statement

A **car company** has released a **new model** and wants to **predict which of their previous customers are most likely to purchase it**. The company has access to historical data that includes:

- **Age** of the customer  
- **Salary** of the customer  
- **Purchase status** (whether they bought a previous model or not)

The goal is to use this data to **predict the probability** that a customer will buy the **new car model**, helping the sales team **target the right audience**.

---

## 🎯 Objective

> Use **Logistic Regression** to build a binary classification model that can predict:
>
> - `1` → Customer will buy the new car  
> - `0` → Customer will not buy the new car  

This enables **personalized marketing** and **efficient resource allocation**.

---

## 🧪 Dataset Example

| Age | Salary (USD) | Purchased |
|-----|--------------|-----------|
| 25  | 50,000       | 0         |
| 45  | 90,000       | 1         |
| 30  | 60,000       | 0         |
| 40  | 80,000       | 1         |
| 35  | 75,000       | 0         |

---

## 🤖 Model Training Workflow

1. **Import Libraries**
   ```python
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.preprocessing import StandardScaler
   from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
   ```

2. **Load and Prepare Data**
   ```python
   # Load dataset
   data = pd.read_csv("customer_data.csv")
   X = data[['Age', 'Salary']]
   y = data['Purchased']
   ```

3. **Train/Test Split**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   ```

4. **Feature Scaling**
   ```python
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)
   ```

5. **Train Logistic Regression**
   ```python
   model = LogisticRegression()
   model.fit(X_train, y_train)
   ```

6. **Make Predictions**
   ```python
   y_pred = model.predict(X_test)
   ```

7. **Evaluate Model**
   ```python
   print("Accuracy:", accuracy_score(y_test, y_pred))
   print(confusion_matrix(y_test, y_pred))
   print(classification_report(y_test, y_pred))
   ```

8. **Predict for a New Customer**
   ```python
   # Example: 38 years old, $85,000 salary
   new_customer = scaler.transform([[38, 85000]])
   prediction = model.predict(new_customer)
   print("Will buy new car:" if prediction[0] == 1 else "Will not buy new car.")
   ```

---

## ✅ Outcome

- 🎯 Identify high-potential buyers
- 📊 Target marketing campaigns more efficiently
- 💰 Increase conversion rate for the new model
- 🤖 Use historical data to inform future sales strategy

---

## 🔧 Notes

- Logistic Regression outputs **probabilities** (e.g., 0.84 = 84% chance of purchase).
- The model works best when features are scaled.
- Consider adding more features (e.g., previous car model, region, marital status) for improved accuracy.

---

## 🧠 Summary

Using **Logistic Regression**, the car company can:
- Predict which previous customers are most likely to buy the new model
- Focus marketing efforts on the right segment
- Improve decision-making with data-backed predictions

