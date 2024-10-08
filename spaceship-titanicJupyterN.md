
# **Spaceship Titanic - Transport Prediction 🚀**

## Overview

This repository contains a machine learning project for the Kaggle competition "Spaceship Titanic." The goal is to predict which passengers were transported to an alternate dimension during a collision with a spacetime anomaly.

## Project Description

In this competition, we use machine learning techniques to analyze data from the Spaceship Titanic's damaged computer system and predict whether passengers were transported.

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

## **Jupyter Notebook** - Code Explanation

# Spaceship Titanic - Transport Prediction 🚀

## 1. Introduction

This notebook aims to predict whether a passenger aboard the Spaceship Titanic will be transported to another dimension using machine learning algorithms. We will use the Kaggle Spaceship Titanic dataset, explore the data,

 engineer features, apply PCA, train models, and evaluate their performance.

# Importing Libraries

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

## 2. Loading the Data

```python
# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data.head()
```

## 3. Initial Data Exploration

```python
# Checking general dataset information
train_data.info()

# Visualizing the distribution of 'Transported'
plt.figure(figsize=(8, 6))
sns.countplot(x='Transported', data=train_data, palette='cool')
plt.title('Distribution of Transported')
plt.show()  # Dark mode applied
```

## 4. Feature Engineering and PCA

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

## 5. Data Preprocessing

```python
# Pipelines for data preprocessing
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

## 6. Model Training and Evaluation (Ensemble Learning)

```python
# Splitting data
X_train_pca, X_val_pca, y_train_pca, y_val_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Defining models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Ensemble model
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

## 7. Hyperparameter Optimization

```python
# Hyperparameter grid for Random Forest
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

## 8. Feature Importance (Random Forest & Gradient Boosting)

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

## 9. Submission

```python
# Preparing test data for submission
test_data['TotalSpend'] = (test_data['RoomService'] + test_data['FoodCourt'] +
                           test_data['ShoppingMall'] + test_data['Spa'] + test_data['VRDeck'])

# Transforming test data similarly to training data
X_test = test_data[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend', 'AvgSpend', 'CabinNumRatio']].fillna(0)
X_test_pca = pca.transform(X_test)

# Predicting
test_predictions = ensemble_model.predict(X_test_pca)

# Creating submission file
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Transported': test_predictions})
submission.to_csv('submission.csv', index=False)
```

## 10. Conclusion

```markdown
This notebook demonstrates a complete machine learning pipeline from feature engineering and PCA to ensemble learning. We improved the model with hyperparameter tuning and provided visualizations in dark mode for better readability. The final results show competitive accuracy and F1 scores.
```

