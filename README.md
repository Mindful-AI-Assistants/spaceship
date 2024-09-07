

# **Spaceship Titanic - Transport Prediction**
## **Predicting Transport on the Spaceship Titanic** - Kaggle Competition

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


## Space Titanic: Analysis and Prediction

This repository contains a project for analyzing and predicting whether passengers will be transported using the Space Titanic dataset from [Kaggle](https://www.kaggle.com/c/spaceship-titanic). The goal is to predict if a passenger will be transported or not.

### 🐍 Code

Aqui está uma versão revisada e aprimorada do código, configurado para uma apresentação elegante e padronizada no formato de um Jupyter Notebook. Abaixo, sugiro como organizar as seções, incluindo melhorias de clareza, consistência e ajustes que facilitam a visualização e interação. Cada célula do notebook será comentada para que sirva como uma apresentação fluída.

---

# **Spaceship Titanic - Transport Prediction**

---

## **1. Introduction**
```markdown
### Introduction
This project aims to predict whether a passenger aboard the Spaceship Titanic will be transported to another dimension using machine learning algorithms. We will use the Kaggle Spaceship Titanic dataset, explore the data, perform feature engineering, and implement various machine learning models to make predictions.
```

---

## **2. Installation of Dependencies**
```python
# Installing necessary libraries
!pip install numpy pandas matplotlib seaborn scikit-learn

# Importing libraries for the project
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

# Ensuring that visualizations are automatically shown
%matplotlib inline
```

---

## **3. Loading the Data**
```python
# Loading the data
train_data = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')

# Checking the first rows of the dataset
train_data.head()
```

---

## **4. Initial Data Exploration**
```python
# Checking general dataset information
train_data.info()

# Checking the count of 'Transported' values
plt.figure(figsize=(8, 6))
sns.countplot(x='Transported', data=train_data)
plt.title('Transported Distribution')
plt.show()
```

---

## **5. Visualization of Numerical Variables and Target**
```python
# Defining numerical features
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Visualizing the relationship between numerical variables and the target variable
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Transported', y=feature, data=train_data)
    plt.title(f'{feature} vs Transported')
    plt.show()
```

---

## **6. Feature Engineering**
```python
# Extracting cabin components
train_data[['Deck', 'Num', 'Side']] = train_data['Cabin'].str.split('/', expand=True)

# Creating a total spend feature
train_data['TotalSpend'] = (train_data['RoomService'] + train_data['FoodCourt'] + 
                            train_data['ShoppingMall'] + train_data['Spa'] + train_data['VRDeck'])

# Checking the first rows after feature engineering
train_data[['Cabin', 'Deck', 'Num', 'Side', 'TotalSpend']].head()
```

---

## **7. Data Preprocessing**
```python
# Creating pipelines for data preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Applying the transformations to the columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, ['HomePlanet', 'Destination', 'Deck', 'Side'])
    ])
```

---

## **8. Data Splitting and Model Training**
```python
# Splitting the data into training and validation sets
X = train_data.drop(columns=['Transported'])
y = train_data['Transported']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the models to be used
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Training and evaluating the models
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

---

## **9. Hyperparameter Optimization**
```python
# Defining the hyperparameters for optimizing Random Forest
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Grid Search for Random Forest
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier())])
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameter combination
print("Best parameters:", grid_search.best_params_)
```

---

## **10. Feature Importance**
```python
# Best model after the search
best_model = grid_search.best_estimator_

# Feature importance
feature_importance = best_model.named_steps['classifier'].feature_importances_
feature_names = np.concatenate([numeric_features, best_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out()])

# Creating a DataFrame for visualization
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(15)

# Visualizing the top 15 important features
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.show()
```

---

## **11. Submission**
```python
# Creating the total spend feature for the test dataset
test_data['TotalSpend'] = (test_data['RoomService'] + test_data['FoodCourt'] + 
                           test_data['ShoppingMall'] + test_data['Spa'] + test_data['VRDeck'])

# Applying the best model to the test data
test_predictions = best_model.predict(test_data)

# Creating the submission file
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Transported': test_predictions})
submission.to_csv('submission.csv', index=False)
```

---

## **Conclusion**
```markdown
### Conclusion
This project demonstrates a complete machine learning pipeline, from data exploration to feature engineering, modeling, and submission for a Kaggle competition. Additionally, we used hyperparameter tuning to enhance the model's performance.
```




<br>


### 🐍 Explaining the key parts of this code and process in more detail:

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




## 📈 Analysis of Results

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
