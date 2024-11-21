# **Spaceship Titanic  🚀 Transport Prediction**


### Starship's flight trajectory 

https://github.com/user-attachments/assets/d307ccdb-a8e4-4548-8c22-563198d6dab9

https://github.com/user-attachments/assets/2a010218-b6a9-468d-97dc-4c6db34271e8

https://github.com/user-attachments/assets/1b82f588-5551-4b17-bd6e-575fbe51e021

<br>



## Overview

This repository contains a machine learning project for the Kaggle competition "Spaceship Titanic." The goal is to predict which passengers were transported to an alternate dimension during a collision with a spacetime anomaly.

## Project Description

In this competition, we use machine learning techniques to analyze data from the Spaceship Titanic's damaged computer system and predict whether passengers were transported.

-----
## Project Structure

1. **Introduction**
2. **Dependencies Installation**
3. **Data Loading**
4. **Initial Data Exploration**
5. **Feature Engineering and PCA**
6. **Data Preprocessing**
7. **Model Training and Evaluation (Ensemble Learning)**
8. **Hyperparameter Optimization**
9. **Feature Importance (Random Forest & Gradient Boosting)**
10. **Submission**
11. **Conclusion**


## 1. Project Structure

1. Project Structure


The project follows a complete machine learning pipeline, which includes:

Installation of Dependencies: Installing and importing necessary Python libraries.

Data Loading: Loading the training and testing datasets.

Exploratory Data Analysis (EDA): A first look at the data through visualization and summary statistics.

Feature Engineering: Enhancing the dataset by creating new variables to improve prediction.

Preprocessing: Handling missing values, scaling numeric features, and encoding categorical variables.

Model Building: Training different machine learning models and evaluating their performance.

Hyperparameter Optimization: Using grid search to fine-tune the best model.

Submission: Predicting on the test set and creating a submission file for Kaggle.



## Getting Started

### Prerequisites

- Python 3.x
- Required Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

### Installation

Install the required libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Usage

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/spaceship-titanic.git
    ```

2. **Navigate to the Project Directory**

    ```bash
    cd spaceship-titanic
    ```

3. **Run the Main Script**

    ```bash
    python main.py
    ```

## Code Explanation

### 1. Introduction

The goal of this project is to predict if a passenger will be transported using machine learning models.

### 2. Installation of Dependencies

```python
!pip install numpy pandas matplotlib seaborn scikit-learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

%matplotlib inline
plt.style.use('dark_background')  # Setting dark mode for visualizations
```

### 3. Loading the Data

```python
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data.head()
```

### 4. Initial Data Exploration

```python
train_data.info()
plt.figure(figsize=(8, 6))
sns.countplot(x='Transported', data=train_data, palette='cool')
plt.title('Distribution of Transported')
plt.show()  # Dark mode applied
```

<br>

Transported  Distribution Graphic

![1-Graf Distrib_Passo4-](https://github.com/user-attachments/assets/4e2755a6-d1bd-4fbb-b68f-0c5accdbe596)

<br>


### 5. Feature Engineering and PCA

```python
# Feature engineering: Total Spend and Average Spend
train_data['TotalSpend'] = train_data['RoomService'] + train_data['FoodCourt'] + train_data['ShoppingMall'] + train_data['Spa'] + train_data['VRDeck']
train_data['AvgSpend'] = train_data['TotalSpend'] / 5
train_data['CabinNumRatio'] = pd.to_numeric(train_data['Num'], errors='coerce') / train_data['Age']

# PCA for dimensionality reduction
X = train_data[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend', 'AvgSpend', 'CabinNumRatio']].fillna(0)
y = train_data['Transported']

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.7)
plt.title('PCA of Features (2 Components) - Dark Mode')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()  # PCA plot in dark mode
```
<br>

PCA of Features (2 Components) Graphic

![2-PCA of Paso5](https://github.com/user-attachments/assets/ac99d3f3-7b3d-4ab2-a1c0-602857f6c667)

<br>

### 6. Data Preprocessing

```python
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']),
        ('cat', categorical_transformer, ['HomePlanet', 'Destination', 'Deck', 'Side'])
    ])
```

### 7. Model Training and Evaluation (Ensemble Learning)

```python
X_train_pca, X_val_pca, y_train_pca, y_val_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('gb', gb_model)], voting='soft')
ensemble_model.fit(X_train_pca, y_train_pca)
y_pred_ensemble = ensemble_model.predict(X_val_pca)

# Metrics
accuracy = accuracy_score(y_val_pca, y_pred_ensemble)
f1 = f1_score(y_val_pca, y_pred_ensemble)
roc_auc = roc_auc_score(y_val_pca, y_pred_ensemble)

print(f"Ensemble Model Accuracy: {accuracy:.4f}")
print(f"Ensemble Model F1 Score: {f1:.4f}")
print(f"Ensemble Model ROC AUC: {roc_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_val_pca, y_pred_ensemble)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title('Confusion Matrix - Ensemble Model (Dark Mode)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

<br>

Confusion Matrix - Random Forest Graphic

![3-Confusion Matrix Passo7](https://github.com/user-attachments/assets/03d45c00-02a0-429b-8ba9-c7ec6462b112)

<br>

### 8. Hyperparameter Optimization

```python
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_pca, y_train_pca)

print("Best parameters:", grid_search.best_params_)
```

### 9. Feature Importance (Random Forest & Gradient Boosting)

```python
ensemble_model.estimators_[0].fit(X_train_pca, y_train_pca)  # Random Forest
feature_importance_rf = ensemble_model.estimators_[0].feature_importances_

ensemble_model.estimators_[1].fit(X_train_pca, y_train_pca)  # Gradient Boosting
feature_importance_gb = ensemble_model.estimators_[1].feature_importances_

importance_df = pd.DataFrame({
    'Feature': ['PC1', 'PC2'],
    'RandomForest': feature_importance_rf,
    'GradientBoosting': feature_importance_gb
})

importance_df = pd.melt(importance_df, id_vars=['Feature'], var_name='Model', value_name='Importance')

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', hue='Model', data=importance_df, palette='coolwarm')
plt.title('Feature Importance by Model (Random Forest vs Gradient Boosting)')
plt.tight_layout()
plt.show()
```
<br> 

Feature Importance by Model (Random Forest vs Gradient Boosting) Graphic

![4-Feature Importance by Model Pass9](https://github.com/user-attachments/assets/c01cf546-60bc-4875-8e1b-2a1cb2e95311)

<br>

### 10. Submission

```python
test_data['TotalSpend'] = (test_data['RoomService'] + test_data['FoodCourt'] +
                           test_data['ShoppingMall'] + test_data['Spa'] + test_data['VRDeck'])

# Assuming you have transformed the test data similarly to the training data
X_test = test_data[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend', 'AvgSpend', 'CabinNumRatio']].fillna(0)
X_test_pca = pca.transform(X_test)

test_predictions = ensemble_model.predict(X_test_pca)

submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Transported': test_predictions})
submission.to_csv('submission.csv', index=False)
```

### 11. Conclusion

```markdown
This project demonstrates a complete machine learning pipeline from feature engineering and PCA to ensemble learning. We further improve the model with hyperparameter tuning and provide visualizations in dark mode for better readability. The final results show competitive accuracy and F1 scores.
```

---

### **Jupyter Notebook**

```python
# Spaceship Titanic - Transport Prediction 🚀

## 1. Introduction

This notebook aims to predict whether a passenger aboard the Spaceship Titanic will be transported to another dimension using machine learning algorithms. We will use the Kaggle Spaceship Titanic dataset, explore the data,
```





'







#


 
##### <p align="center">Copyright 2024 Mindful-AI-Assistants. Code released under the  [MIT license.]( https://github.com/Mindful-AI-Assistants/.github/blob/ad6948fdec771e022d49cd96f99024fcc7f1106a/LICENSE)

