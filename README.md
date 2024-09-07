

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


## Space Titanic: Analysis and Prediction

This repository contains a project for analyzing and predicting whether passengers will be transported using the Space Titanic dataset from [Kaggle](https://www.kaggle.com/c/spaceship-titanic). The goal is to predict if a passenger will be transported or not.

### 🐍 Code







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
