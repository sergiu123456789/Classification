# 🚗 Support Vector Machine (SVM) Problem: Predicting Car Purchase Behavior

## 🧩 Problem Statement

A **car company** is launching a **new model** and wants to target previous customers who are **most likely to buy** the new car. To do this, the company wants to use **machine learning** to predict whether a customer will buy the car, based on their:

- **Age**
- **Salary**

This is a **binary classification problem**, where the goal is to predict whether a customer will **purchase** (1) or **not purchase** (0) the new car model.

---

## 🎯 Objective

> Build a **classification model** using **Support Vector Machine (SVM)** with an **RBF (Radial Basis Function) kernel** to predict car purchases based on customer age and salary.

---

## 🧪 Dataset Example

| Customer ID | Age | Salary (USD) | Purchased |
|-------------|-----|--------------|-----------|
| 1           | 25  | 50,000       | 0         |
| 2           | 30  | 60,000       | 1         |
| 3           | 22  | 35,000       | 0         |
| 4           | 35  | 80,000       | 1         |
| 5           | 45  | 120,000      | 1         |

---

## ⚙️ Why SVM with RBF Kernel?

- **SVM** is effective for high-dimensional spaces and works well with **small to medium datasets**.
- The **RBF kernel** captures **non-linear relationships** between features (age and salary) and the target label.
- Helps find the **optimal decision boundary** even when the data is not linearly separable.

---

## 📘 Model Building Steps

1. **Import libraries**:
   ```python
   from sklearn.svm import SVC
   from sklearn.preprocessing import StandardScaler
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score, confusion_matrix
   ```

2. **Prepare the data**:
   ```python
   X = dataset[['Age', 'Salary']].values
   y = dataset['Purchased'].values
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.transform(X_test)
   ```

3. **Train the SVM model**:
   ```python
   classifier = SVC(kernel='rbf', random_state=0)
   classifier.fit(X_train, y_train)
   ```

4. **Make predictions and evaluate**:
   ```python
   y_pred = classifier.predict(X_test)

   accuracy = accuracy_score(y_test, y_pred)
   cm = confusion_matrix(y_test, y_pred)
   print(f"Accuracy: {accuracy}")
   print(f"Confusion Matrix:\n{cm}")
   ```

---

## ✅ Outcome

- Predict **which customers are most likely to buy** the new car model
- Improve **marketing campaign targeting**
- Increase **conversion rates** and **reduce costs** by avoiding outreach to uninterested customers

---

## 🧠 Summary

Using **SVM with RBF kernel**, the car company can:

- Accurately classify potential buyers based on demographic data
- Leverage non-linear patterns in purchasing behavior
- Make smart, data-driven business decisions

📌 This approach enables personalized marketing and better ROI on customer outreach for new product launches.

