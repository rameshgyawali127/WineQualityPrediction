# Wine-Quality-Prediction
Objective:
   The Main objective of this code is to perform a comprehensive analysis on a wine quality dataset. It includes data preprocessing, exploratory data analysis (EDA), visualization, and the evaluation of three 
   different classification models (Logistic Regression, XGBoost Classifier, and Support Vector Classifier) to predict wine quality based on various features. The code aims to provide insights into the dataset, 
   handle missing data, visualize key relationships, and determine the best-performing model for quality prediction while considering metrics such as ROC AUC, confusion matrix, and classification report.

Algorithm Used in this Project are:
Logistic Regression: Used for binary classification to predict whether a wine quality is above a certain threshold (quality > 5).

Support Vector Classifier (SVC): Utilized with the Radial Basis Function (RBF) kernel for classification tasks.

XGBoost Classifier: A gradient boosting algorithm used for classification tasks. It's an ensemble technique that combines multiple decision trees.

Key Project Steps:

Data Exploration and Preprocessing:

The dataset is loaded and examined for insights into data types and missing values.
Missing values are imputed with column means, ensuring data completeness.
Data distributions are visualized through histograms and correlation analysis.
Feature Engineering:

The 'total sulfur dioxide' column is dropped as it is deemed unnecessary.
A new binary column, 'best quality,' is created to represent quality categories.
The 'type' column is transformed for modeling.
Data Splitting and Normalization:

The dataset is split into training and testing sets (80% train, 20% test).
Min-Max scaling is applied to normalize feature values.
Model Selection and Evaluation:

Three classification models, including Logistic Regression, XGBoost, and SVC with RBF kernel, are selected.
Models are trained and evaluated using ROC AUC scores on both training and testing sets, assessing their predictive accuracy.
Model Performance Visualization:

Confusion matrices are plotted, providing visual insights into model performance, with a focus on XGBoost.
Classification reports, encompassing precision, recall, and F1-score, are generated for detailed model assessment.


In conclusion, this wine quality prediction project leveraged a combination of machine learning algorithms to determine whether a wine's quality exceeds a predefined threshold. Logistic Regression, Support Vector Classifier with RBF kernel, and XGBoost Classifier were employed to achieve this task. The project encompassed thorough data exploration, preprocessing, and model evaluation, culminating in the successful prediction of wine quality based on various features. Through comprehensive visualizations and performance metrics, the project provides valuable insights into wine quality classification, demonstrating the effectiveness of these algorithms in addressing the problem at hand.
   
