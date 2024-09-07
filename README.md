

# 🚀 Spaceship Titanic Prediction - Kaggle Competition
#### This competition runs indefinitely with a rolling leaderboard. [Learn more](https://www.kaggle.com/docs/competitions#getting-started).


## Welcome to the Spaceship 

Titanic repository! 

This project is an entry for the Kaggle competition "Spaceship Titanic", where we predict which passengers were transported to an alternate dimension during the Spaceship Titanic's collision with a spacetime anomaly.


## 📊 Project Description
In this competition, we use machine learning techniques to analyze the recovered records from the Spaceship Titanic's damaged computer system and predict which passengers were transported to an alternate dimension.




1. A project description
2. Instructions for getting started
3. An overview of the project structure
4. A brief explanation of the methodology
5. Placeholders for results and future work
6. Sections on contributing and licensing
7. Acknowledgments

## Dataset

The dataset includes passenger information such as:
- Age
- Home Planet
- Cabin
- Destination
- VIP status
- Expenditures on various amenities

## Getting Started

### Prerequisites

To run this project, you need to have Python installed on your system. The code is written in Python 3.

### Required Libraries

Install the required libraries using pip:

```
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Running the Code

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/spaceship-titanic.git
   ```
2. Navigate to the project directory:
   ```
   cd spaceship-titanic
   ```
3. Run the main script:
   ```
   python main.py
   ```

## Project Structure

- `main.py`: The main script that runs the entire pipeline
- `data_preprocessing.py`: Functions for data cleaning and feature engineering
- `model.py`: Contains the machine learning model and related functions
- `utils.py`: Utility functions used across the project

## Methodology

1. Data Preprocessing: Cleaning the data, handling missing values, and feature engineering
2. Exploratory Data Analysis: Visualizing the data to gain insights
3. Model Selection: Trying various models and selecting the best performer
4. Hyperparameter Tuning: Optimizing the model parameters
5. Prediction: Generating predictions for the test set



## Spaceship Titanic Analysis and Modeling

### 🐍 Code

# Space Titanic: Analysis and Prediction

This repository contains a project for analyzing and predicting whether passengers will be transported using the Space Titanic dataset from [Kaggle](https://www.kaggle.com/c/spaceship-titanic). The goal is to predict if a passenger will be transported or not.


#### 1. Project Title
```markdown
# Spaceship Titanic - Transport Prediction
```

#### 2. Project Description
```markdown
This project aims to predict whether a passenger aboard the Spaceship Titanic will be transported to another dimension. We use the dataset available in the Kaggle Spaceship Titanic competition and apply various Machine Learning techniques such as Logistic Regression, Random Forest, and Gradient Boosting to make predictions.
```

#### 3. Installation
```markdown
### Prerequisites
Ensure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install the dependencies by running the following command:
```bash
pip install -r requirements.txt
```

#### 4. Loading Data
```markdown
### Loading Data
You can load the data directly from Kaggle:

```python
import pandas as pd

train_data = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
```
```

#### 5. Data Exploration
```markdown
### Data Exploration

In this step, we explore the features of the dataset. Below are some initial visualizations:

- Transported count distribution:
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.countplot(x='Transported', data=train_data)
plt.title('Transported Distribution')
plt.show()
```

- Relationship between numerical variables and the target (Transported):
```python
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Transported', y=feature, data=train_data)
    plt.title(f'{feature} vs Transported')
    plt.show()
```
```

#### 6. Feature Engineering
```markdown
### Feature Engineering

We create new features, such as extracting components from the cabin information and creating a "total spend" column:

```python
# Extract components from cabin
train_data[['Deck', 'Num', 'Side']] = train_data['Cabin'].str.split('/', expand=True)

# Create a feature for total spend
train_data['TotalSpend'] = (train_data['RoomService'] + train_data['FoodCourt'] + 
                            train_data['ShoppingMall'] + train_data['Spa'] + train_data['VRDeck'])
```
```

#### 7. Data Preprocessing
```markdown
### Data Preprocessing

We apply pipelines to handle missing values, normalize numerical features, and encode categorical variables:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Pipeline for numerical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Apply to full dataset
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, ['HomePlanet', 'Destination', 'Deck', 'Side'])
    ])
```
```

#### 8. Model Training and Evaluation
```markdown
### Model Training and Evaluation

We use three models: Logistic Regression, Random Forest, and Gradient Boosting, and evaluate their accuracy using a validation set.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Splitting data
X = train_data.drop(columns=['Transported'])
y = train_data['Transported']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

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
    plt.show()
```
```

#### 9. Hyperparameter Tuning
```markdown
### Hyperparameter Tuning

We tune the hyperparameters of the Random Forest model:

```python
from sklearn.model_selection import GridSearchCV

# Hyperparameter grid
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best parameters:", grid_search.best_params_)
```
```

#### 10. Feature Importance
```markdown
### Feature Importance

We analyze the feature importance for the tuned Random Forest model:

```python
# Best model
best_model = grid_search.best_estimator_

# Feature importance
feature_importance = best_model.named_steps['classifier'].feature_importances_
feature_names = preprocessor.transformers_[0][2] + preprocessor.transformers_[1][1].get_feature_names_out()

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(15)

# Plot top 15 important features
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.show()
```
```

#### 11. Submission
```markdown
### Submission

Finally, we apply the trained model on the test set and create the submission file for the Kaggle competition.

```python
test_data['TotalSpend'] = (test_data['RoomService'] + test_data['FoodCourt'] + 
                           test_data['ShoppingMall'] + test_data['Spa'] + test_data['VRDeck'])

# Apply the model to the test set
test_predictions = best_model.predict(test_data)

# Create submission file
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Transported': test_predictions})
submission.to_csv('submission.csv', index=False)
```
```

---

### Enabling Automatic Visualizations in VSCode for Jupyter Notebooks:

- When using VSCode with the Jupyter Notebook extension, ensure that the interactive mode for Matplotlib is enabled by setting `matplotlib inline` at the top of the notebook:

```python
%matplotlib inline
```

- This ensures that all visualizations generated with `plt.show()` will be displayed automatically in the interactive notebook environment.










'
---0----
## Dependencies

The code uses the following Python libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
```

## Loading the Data

```python
# Load data
train_data = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
```

## Data Exploration

### Basic Information

```python
# Display basic information about the dataset
print(train_data.info())

# Check for missing values
print(train_data.isnull().sum())
```

### Visualizations

#### Distribution of Target Variable

```python
plt.figure(figsize=(8, 6))
sns.countplot(x='Transported', data=train_data)
plt.title('Distribution of Transported')
plt.show()
```

#### Numeric Features vs Target Variable

```python
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Transported', y=feature, data=train_data)
    plt.title(f'{feature} vs Transported')
    plt.show()
```

#### Categorical Features vs Target Variable

```python
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, hue='Transported', data=train_data)
    plt.title(f'{feature} vs Transported')
    plt.xticks(rotation=45)
    plt.show()
```

## Feature Engineering

```python
# Extract deck, num, and side from Cabin
train_data[['Deck', 'Num', 'Side']] = train_data['Cabin'].str.split('/', expand=True)
test_data[['Deck', 'Num', 'Side']] = test_data['Cabin'].str.split('/', expand=True)

# Create a feature for total spending
train_data['TotalSpend'] = train_data['RoomService'] + train_data['FoodCourt'] + train_data['ShoppingMall'] + train_data['Spa'] + train_data['VRDeck']
test_data['TotalSpend'] = test_data['RoomService'] + train_data['FoodCourt'] + train_data['ShoppingMall'] + train_data['Spa'] + train_data['VRDeck']

# Create a feature for group size
train_data['GroupSize'] = train_data.groupby('PassengerId').transform('size')
test_data['GroupSize'] = test_data.groupby('PassengerId').transform('size')
```

## Data Preprocessing

```python
# Define features
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend', 'GroupSize']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

# Create preprocessing pipelines
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
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Prepare the data
X = train_data.drop(['PassengerId', 'Transported', 'Name', 'Cabin', 'Num'], axis=1)
y = train_data['Transported']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Model Selection and Training

```python
# Define models to try
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Train and evaluate models
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    
    print(f"\n{name}:")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
```

## Hyperparameter Tuning

```python
# Example: Tuning Random Forest
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier())])

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
print("\nBest Model Performance:")
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))
```

## Feature Importance

```python
# Get feature importance from the best Random Forest model
feature_importance = best_model.named_steps['classifier'].feature_importances_
feature_names = (best_model.named_steps['preprocessor']
                 .named_transformers_['cat']
                 .named_steps['onehot']
                 .get_feature_names(categorical_features).tolist() + numeric_features)

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(15)
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.show()
```

## Preparing Submission

```python
# Preprocess test data
X_test = test_data.drop(['PassengerId', 'Name', 'Cabin', 'Num'], axis=1)
X_test_processed = best_model.named_steps['preprocessor'].transform(X_test)

# Make predictions
test_predictions = best_model.named_steps['classifier'].predict(X_test_processed)

# Create submission DataFrame
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Transported': test_predictions})
submission['Transported'] = submission['Transported'].map({False: False, True: True})

# Save submission file
submission.to_csv('submission.csv', index=False)
print("Submission file created.")
```

### Explaining the key parts of this code and process in more detail:

1. ### Data Exploration
   - We start by loading the data and examining its structure using `train_data.info()`.
   - We check for missing values using `train_data.isnull().sum()`.
   - We visualize the distribution of the target variable ('Transported') and explore its relationship with numeric and categorical features using seaborn plots.

2. ### Feature Engineering
   - We extract information from the 'Cabin' column, splitting it into 'Deck', 'Num', and 'Side'.
   - We create a 'TotalSpend' feature by summing up all the spending-related columns.
   - We add a 'GroupSize' feature by grouping passengers with similar IDs.

3. ### Preprocessing
   - We define separate pipelines for numeric and categorical features.
   - For numeric features, we impute missing values with the median and apply standard scaling.
   - For categorical features, we impute missing values with 'missing' and apply one-hot encoding.
   - We use `ColumnTransformer` to apply these preprocessing steps to the appropriate columns.

4. ### Model Selection and Training
   - We try three different models: Logistic Regression, Random Forest, and Gradient Boosting.
   - For each model, we create a pipeline that includes the preprocessor and the classifier.
   - We train each model and evaluate its performance using accuracy score and classification report.
   - We visualize the confusion matrix for each model.

5. ### Hyperparameter Tuning
   - We perform a grid search for the Random Forest model to find the best hyperparameters.
   - We use cross-validation to ensure robust results.
   - After finding the best parameters, we evaluate the best model on the validation set.

6. ### Feature Importance
   - We extract feature importance from the best Random Forest model.
   - We visualize the top 15 most important features.

7. ### Prepare Submission
   - We preprocess the test data using the same pipeline.
   - We make predictions on the test set using the best model.
   - We create a submission file in the format required by Kaggle.

This process allows for a comprehensive exploration of the data, careful feature engineering, and a systematic approach to model selection and improvement. The use of pipelines ensures that all preprocessing steps are consistently applied to both training and test data.

### Analysis of Results

1. **Distribution of Transported**:
   - **Plot**: Shows the count of transported vs. non-transported passengers. This helps understand the class distribution.

2. **Numeric Features vs. Target**:
   - **Plots**: Box plots for numeric features such as `Age`, `RoomService`, etc., against the target variable `Transported`. These plots reveal the distribution of numerical features based on the transportation status.

3. **Categorical Features vs. Target**:
   - **Plots**: Count plots for categorical features like `HomePlanet`, `CryoSleep`, etc., showing how these features relate to the target variable.

4. **Confusion Matrix**:
   - **Plots**: Heatmaps of confusion matrices for each model, providing insight into the true positive, false positive, true negative, and false negative predictions.

5. **Feature Importance**:
   - **Plot**: A bar plot of the top 15 most important features from the best-performing model. This helps identify which features have the most influence on predictions.

These plots and analyses help to understand the data, the performance of various models, and the significance of different features in the prediction task.




## 👀 Future Work

- Try ensemble methods to improve prediction accuracy
- Explore deep learning approaches
- Conduct more in-depth feature engineering

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.


🙏 Acknowledgments
Kaggle for hosting the competition and providing the dataset
The Scikit-learn team for their excellent machine learning library

#

This project is licensed under the MIT License - see the LICENSE.md file for details.
