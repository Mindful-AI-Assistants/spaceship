Aqui está o código revisado e o README.md formatado de maneira profissional, com um layout claro para o GitHub e o Jupyter Notebook, incluindo as seções onde os gráficos gerados devem ser inseridos.

---

### **Versão Final para GitHub README.md**:

```markdown
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
5. **Feature Engineering**
6. **Data Preprocessing**
7. **Model Training and Evaluation**
8. **Hyperparameter Optimization**
9. **Feature Importance**
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

## Code Explanation

### 1. Introduction

The goal of this project is to predict if a passenger will be transported using machine learning models.

### 2. Installation of Dependencies

```python
# Installing necessary libraries
!pip install numpy pandas matplotlib seaborn scikit-learn

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Ensuring visualizations are displayed inline
%matplotlib inline
```

### 3. Loading the Data

```python
# Loading the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Displaying the first few rows of the dataset
train_data.head()
```

### 4. Initial Data Exploration

```python
# Checking general dataset information
train_data.info()

# Plotting distribution of 'Transported'
plt.figure(figsize=(8, 6))
sns.countplot(x='Transported', data=train_data)
plt.title('Distribution of Transported')
plt.show()  # Insert the generated plot here
```

### 5. Feature Engineering

```python
# Extracting cabin components
train_data[['Deck', 'Num', 'Side']] = train_data['Cabin'].str.split('/', expand=True)

# Creating a total spend feature
train_data['TotalSpend'] = (train_data['RoomService'] + train_data['FoodCourt'] +
                            train_data['ShoppingMall'] + train_data['Spa'] + train_data['VRDeck'])

# Displaying feature engineering results
train_data[['Cabin', 'Deck', 'Num', 'Side', 'TotalSpend']].head()
```

### 6. Data Preprocessing

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

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']),
        ('cat', categorical_transformer, ['HomePlanet', 'Destination', 'Deck', 'Side'])
    ])
```

### 7. Model Training and Evaluation

```python
# Splitting data into training and validation sets
X = train_data.drop(columns=['Transported'])
y = train_data['Transported']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Training and evaluating models
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    
    print(f"\n{name} - Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()  # Insert the generated plot here
```

### 8. Hyperparameter Optimization

```python
# Hyperparameter grid for Random Forest
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Grid search for Random Forest
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier())])
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters:", grid_search.best_params_)
```

### 9. Feature Importance

```python
# Best model after grid search
best_model = grid_search.best_estimator_

# Feature importance
feature_importance = best_model.named_steps['classifier'].feature_importances_
feature_names = np.concatenate([['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'],
                                  best_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out()])

# Creating and visualizing feature importance DataFrame
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.show()  # Insert the generated plot here
```

### 10. Submission

```python
# Creating total spend feature for test data
test_data['TotalSpend'] = (test_data['RoomService'] + test_data['FoodCourt'] +
                           test_data['ShoppingMall'] + test_data['Spa'] + test_data['VRDeck'])

# Applying the best model to test data
test_predictions = best_model.predict(test_data)

# Creating and saving the submission file
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 
                           'Transported': test_predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")
```

### 11. Conclusion

```markdown
This project demonstrates a complete machine learning pipeline from data exploration, feature engineering, and model training to hyperparameter tuning and submission. The pipeline ensures thorough analysis and optimization to achieve the best model performance.
```

---

### **Versão Finalizada para Jupyter Notebook**:

O conteúdo da versão Jupyter Notebook já está adequado, e pode ser diretamente adaptado para o formato do Jupyter, como descrito no exemplo anterior. O código pode ser executado sequencialmente, e as instruções sobre onde inserir os gráficos estão devidamente comentadas nas seções correspondentes.

---

Agora você tem um README.md elegante para o GitHub, juntamente com um código Python claro e profissional para execução no Jupyter Notebook!
