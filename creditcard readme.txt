Credit Card Approval Prediction
This project is designed to predict credit card approval based on various features, including age, income, credit score, and marital status, using machine learning models. The dataset used is synthetic, created for testing purposes. The project covers data preprocessing, handling class imbalance, training multiple models, and evaluating their performance.

Features
Data Creation: Generates a synthetic dataset for testing.
Data Preprocessing: Handles missing values, encodes categorical data, and scales numerical data.
Model Training: Trains multiple models, including Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, and XGBoost.
Class Imbalance Handling: Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes.
Model Evaluation: Evaluates models on test data and prints classification reports.
Feature Importance: Analyzes and visualizes feature importance for tree-based models.
Prerequisites
Ensure you have the following Python libraries installed:

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imblearn
You can install them using pip:

bash
Copy
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
Code Walkthrough
1. Dataset Creation
If the dataset does not exist, a synthetic dataset is created with columns such as:

Age
Income
LoanAmount
CreditScore
MaritalStatus
Gender
Approval (target column)
This dataset is saved to a CSV file for later use.

2. Data Preprocessing
Missing Values: The function handles missing values by filling them with median for numerical columns and mode for categorical columns.
Label Encoding: Converts categorical columns (Gender, MaritalStatus) into numeric values using LabelEncoder.
Feature Scaling: Standardizes the numerical columns using StandardScaler.
3. Model Training
Trains the following models:
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Gradient Boosting Classifier
XGBoost Classifier
All models are trained using the training data (X_train and y_train).
4. Class Imbalance Handling
Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset by generating synthetic samples for the minority class.
5. Model Evaluation
Classification Report: Prints precision, recall, f1-score, and support for each model.
Confusion Matrix: Shows the confusion matrix for each model.
Accuracy Score: Computes the accuracy score for each model.
6. Feature Importance
For tree-based models (Random Forest, Decision Tree, Gradient Boosting, XGBoost), the function calculates and plots feature importance to understand which features contribute the most to the predictions.
How to Use
Run the Script: After setting up the environment with the required libraries, you can run the script. The script will:

Check if the dataset exists, and if not, it will create it.
Preprocess the data, handle class imbalance, and split the data into training and testing sets.
Train and evaluate several machine learning models.
Display classification reports, confusion matrices, and accuracy scores.
Show feature importance plots for tree-based models.
To run the script, use the following command in your terminal or command prompt:

bash
Copy
python credit_card_approval_prediction.py
Output: The script will output the following:

Classification Reports: Precision, recall, f1-score, and support for each model.
Confusion Matrices: For each model.
Accuracy: The accuracy score of each model.
Feature Importance: Visualizations of feature importance for tree-based models.
Troubleshooting
Dataset Not Found: If the dataset is missing, the script will automatically create a synthetic dataset. Ensure that the file path is correct.
Library Issues: Ensure that all the required libraries (pandas, numpy, scikit-learn, etc.) are installed via pip as mentioned in the prerequisites section.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
scikit-learn for the machine learning algorithms.
XGBoost for the XGBoost implementation.
imbalanced-learn for SMOTE implementation.
matplotlib and seaborn for visualizations.
