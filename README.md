# Predicting Heart Disease - A Comprehensive Analysis and Model Building Report

### by Tendai Milicent Jonhasi

## Overview
This project aims to develop a predictive model to classify individuals as having or not having heart disease based on various patient attributes, including demographic, clinical, and lifestyle factors. The ultimate goal is to improve early detection of heart disease, contributing to preventive healthcare.

## Project Objectives
1. Perform Exploratory Data Analysis (EDA) to understand the dataset and identify patterns.
2. Preprocess the data by handling missing values and scaling features.
3. Build multiple predictive models for heart disease classification.
4. Evaluate model performance using accuracy, precision, recall, and AUC-ROC metrics.
5. Select and interpret the best-performing model.

## Dataset
The dataset includes the following features:
- Demographic attributes: Age, Gender, etc.
- Clinical features: Cholesterol level, Blood pressure, Glucose, etc.
- Lifestyle factors: Smoking, Alcohol consumption, Physical activity, etc.

## Key Libraries Used
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning model building and evaluation.
- **XGBoost, Random Forest, Logistic Regression**: For predictive modeling.

## Steps Involved
1. **Data Loading and Preprocessing**: 
    - Cleaned and prepared the dataset by handling missing values and encoding categorical variables.
    - Scaled numerical features for consistent model training.

2. **Exploratory Data Analysis (EDA)**:
    - Visualized the relationships between features and the target variable.
    - Identified key features that influence the prediction of heart disease.

3. **Model Building**:
    - Built multiple models, including Logistic Regression, Random Forest, and XGBoost classifiers.
    - Trained and tuned the models using cross-validation and hyperparameter optimization.
    - - *Logistic Regression* -- Accuracy: 71,8% • Key Strengths: High recall for non-heart disease cases (76%)
• Weaknesses: Lower recall for heart disease cases (68%)
• Age, cholesterol, blood pressure, and weight are significant predictors of heart disease.
    - - *Random Forest* -- Accuracy: 72.5%
• Key Strengths: Balanced recall for both non-heart disease (75%) and heart disease cases (70%), offering a robust model for identifying at-risk patients.
• Age, systolic blood pressure, and weight are the most influential factors.    
    - - *Support Vector Machine (SVM)* -- Accuracy: 72.3%
• Key Strengths: Effective in high-dimensional spaces and good at identifying non-heart disease cases.
• Weaknesses: Lower recall for heart disease cases (68%), which could lead to underdiagnosis.
• Blood pressure, cholesterol, and age are the most significant factors in the model.     
    - - *Convolutional Neural Network* -- Accuracy: 76%
• Key Strengths: Best performance among all models with the highest accuracy and AUC-ROC, indicating strong predictive power.
• Weaknesses: While effective overall, CNNs are less interpretable compared to traditional models. Unlike other models, CNNs automatically learn to extract features, which may include the same key factors (age, blood pressure, weight) but are less transparent.

4. **Model Evaluation**:
    - Evaluated the models using classification metrics such as accuracy, precision, recall, F1-score, and AUC-ROC curve.
    - Compared the performance of different models and selected the best-performing one based on the evaluation metrics.

## Results
The best-performing model was selected based on its ability to balance precision and recall, with a particular focus on minimizing false negatives to ensure fewer cases of undetected heart disease.

**Recommendations**:
1. Suggested Model: Convolutional Neural Network (CNN)
The CNN outperforms the other models, achieving the highest accuracy (76%) and AUC-ROC score (0.80).It provides a strong predictive capability for distinguishing between individuals with and without heart disease.
2. Second-Best Model: Random Forest
The Random Forest model offers a good balance between recall for both classes, making it a reliable choice when interpretability and feature importance are critical.
3. Feature Importance & Coefficient Impact: Logistic Regression and Random Forest models are recommended for understanding feature importance. The insights from these models can guide targeted interventions and personalized care, focusing on managing age, blood pressure, cholesterol levels, and weight to mitigate the risk of heart disease.

## Conclusion
In conclusion, the models showed that individuals who are older, have high blood pressure, high cholesterol, are overweight or obese, and have unhealthy lifestyle habits like smoking, excessive alcohol consumption, and low physical activity are at the highest risk of developing heart disease. These factors should be closely monitored and managed to reduce the risk of heart disease, and targeted interventions should be considered for those at high risk.
This project successfully demonstrates how machine learning can be applied to healthcare data to predict the presence of heart disease with high accuracy. The models built in this project provide a foundation for further improvements and practical applications in medical diagnostics.

## Installation & Usage
To replicate this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
