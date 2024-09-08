

'
🛸　　　 　🌎　°　　🌓　•　　.°•　　　🚀 ✯ 🛸　.　　•.  　🌎　°　.•　🌓　•　　.°•　　•　🚀 ✯.    •.    .  •. 
　.　•　★　*　　　　　°　　.　　🛰 　°·　　•.      ๏        .•       🪐  
.　•　•　° ★　•  ☄.       ๏       •.      .  •.      .     •.      .  🛸　.　　•.  　🌎　°　.•　🌓　•　　.°•　　•　🚀 ✯.    •.    .  •. 
　.　•　★　*　　　　　°　　.　　🛰 　°·　　•.      ๏        .•       🪐
.　•　•　° ★　•  ☄.       ๏       •.      .  •.      .     •.      .     •. •    
　　　★　*　　　　　°　　　　🛰 　°·　👩‍🚀🚀⋆ ⭒˚.⋆ 🪐  ⋆⭒˚.⋆                
.　　　•　° ★　•  ☄


<br><br>


# **Spaceship Titanic - Transport Prediction**

This project aims to predict whether a passenger aboard the Spaceship Titanic will be transported to another dimension using machine learning algorithms. We'll use the dataset provided by Kaggle, explore the data, perform feature engineering, and implement various machine learning models to predict the outcome with high accuracy.



## **1. Project Structure**

The project follows a complete machine learning pipeline, which includes:

1. **Installation of Dependencies**: Installing and importing necessary Python libraries.
2. **Data Loading**: Loading the training and testing datasets.
3. **Exploratory Data Analysis (EDA)**: A first look at the data through visualization and summary statistics.
4. **Feature Engineering**: Enhancing the dataset by creating new variables to improve prediction.
5. **Preprocessing**: Handling missing values, scaling numeric features, and encoding categorical variables.
6. **Model Building**: Training different machine learning models and evaluating their performance.
7. **Hyperparameter Optimization**: Using grid search to fine-tune the best model.
8. **Submission**: Predicting on the test set and creating a submission file for Kaggle.



## **2. Installation of Dependencies**

First, make sure to install all the required libraries for the project. Run the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Then, import the necessary libraries:

```python
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

# Ensuring that visualizations are automatically shown
%matplotlib inline
```

---

## **3. Loading the Data**

Load the dataset into a pandas DataFrame:

```python
# Load the training and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Check the first few rows of the training data
train_data.head()
```

---

## **4. Exploratory Data Analysis (EDA)**

Let's start by exploring the data. We can check the distribution of the target variable (`Transported`) and investigate the data types.

```python
# Checking general dataset information
train_data.info()

# Checking the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='Transported', data=train_data)
plt.title('Transported Distribution')
plt.show()
```

Next, visualize the relationship between the numerical features and the target variable:

```python
# Define numerical features
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Visualizing the relationship between numerical variables and the target
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Transported', y=feature, data=train_data)
    plt.title(f'{feature} vs Transported')
    plt.show()
```



## **5. Feature Engineering**

Create new features that might help improve the model’s performance, such as splitting the `Cabin` column and calculating total spending.

```python
# Split the Cabin column into Deck, Number, and Side
train_data[['Deck', 'Num', 'Side']] = train_data['Cabin'].str.split('/', expand=True)

# Create a new feature representing the total spend of a passenger
train_data['TotalSpend'] = (train_data['RoomService'] + train_data['FoodCourt'] + 
                            train_data['ShoppingMall'] + train_data['Spa'] + train_data['VRDeck'])

# Check the new features
train_data[['Cabin', 'Deck', 'Num', 'Side', 'TotalSpend']].head()
```



## **6. Data Preprocessing**

Preprocessing the data is crucial to ensure that machine learning models can interpret the features correctly. We will impute missing values, scale the numeric data, and encode categorical features.

```python
# Pipeline for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine the pipelines into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, ['HomePlanet', 'Destination', 'Deck', 'Side'])
    ])
```

---

## **7. Model Building**

We will train three different machine learning models: Logistic Regression, Random Forest, and Gradient Boosting. After training each model, we will evaluate its performance.

```python
# Splitting the data into training and validation sets
X = train_data.drop(columns=['Transported'])
y = train_data['Transported']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    
    print(f"\n{name} - Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    
    # Display the confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()
```

---

## **8. Hyperparameter Optimization**

We'll optimize the Random Forest model using GridSearchCV to find the best combination of hyperparameters.

```python
# Define the hyperparameters grid
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Perform grid search for Random Forest
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier())])
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Output the best parameters
print("Best parameters:", grid_search.best_params_)
```

---

## **9. Feature Importance**

After finding the best model, we can look at the most important features used by the model.

```python
# Get the best model
best_model = grid_search.best_estimator_

# Feature importance from the Random Forest model
feature_importance = best_model.named_steps['classifier'].feature_importances_
feature_names = np.concatenate([numeric_features, best_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out()])

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(15)

# Plot the top 15 most important features
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.show()
```


## **10. Submission**

Finally, we use the best model to predict the test set and create the submission file for Kaggle.

```python
# Apply the feature engineering to the test data
test_data['TotalSpend'] = (test_data['RoomService'] + test_data['FoodCourt'] + 
                           test_data['ShoppingMall'] + test_data['Spa'] + test_data['VRDeck'])

# Predict on the test set
test_predictions = best_model.predict(test_data)

# Create the submission file
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Transported': test_predictions})
submission.to_csv('submission.csv', index=False)

# Final output message
print("Submission file 'submission.csv' created successfully.")
```


## **11. Conclusion**

This project illustrates the complete machine learning workflow, from data exploration to feature engineering, model building, hyperparameter tuning, and submission. By employing various models and optimizing them through grid search, we achieved better predictions on the Spaceship Titanic dataset.

For more detailed exploration, you can access the [complete Jupyter notebook](link-to-notebook) with all the code and visualizations.


## Key Notes:

Certainly! Continuing from where we left off:

```python
# Feature Importance (cont'd)
feature_names = np.concatenate([numeric_features, 
                                  best_model.named_steps['preprocessor']
                                  .transformers_[1][1]
                                  .get_feature_names_out()])

# Creating a DataFrame for feature importance
feature_importance_df = pd.DataFrame({'feature': feature_names, 
                                      'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='importance', 
                                                          ascending=False).head(15)

# Visualizing feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.show()

# 12. Making Predictions on the Test Data
test_data['TotalSpend'] = (test_data['RoomService'] + test_data['FoodCourt'] +
                           test_data['ShoppingMall'] + test_data['Spa'] + test_data['VRDeck'])

# Applying the best model to the test data
test_predictions = best_model.predict(test_data)

# Creating the submission file
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 
                           'Transported': test_predictions})
submission.to_csv('submission.csv', index=False)

# Final output message
print("Submission file 'submission.csv' created successfully.")
```


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
