📄 Auto ML Prediction Report
Project Overview

This report summarizes the outcomes of an automated machine learning process using a custom Streamlit web app. The application allows business users to:

Upload any structured dataset (CSV format)

Automatically clean and preprocess data

Select a target variable and appropriate ML model

Train and evaluate predictive models with no code required

🧼 Data Handling & Cleaning
Columns with over 50% missing values are removed

Continuous features with missing values are filled with the mean

Categorical features are filled with the mode

All categorical variables are one-hot encoded

🎯 Target Variable
The user selects a target column which the model is trained to predict. The app automatically detects whether the prediction is a regression (numeric target) or classification (categorical target) problem.

⚙️ Models Available
Classification Models

Logistic Regression

Random Forest Classifier

Gradient Boosting Classifier

AdaBoost Classifier

KNeighbors Classifier

Gaussian Naive Bayes

Decision Tree Classifier

Regression Models

Linear Regression

Random Forest Regressor

Gradient Boosting Regressor

AdaBoost Regressor

KNeighbors Regressor

Decision Tree Regressor

📊 Evaluation Metrics
Classification

Accuracy, Precision, Recall, F1-Score per class

Macro and Weighted Averages

Regression

R² Score

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

📈 Test Set Predictions
The application also displays side-by-side comparisons of actual vs. predicted values on the test set.

💡 Business Implications
Speed: Reduces time-to-insight by enabling fast prototyping and model training without needing a data scientist.

Accessibility: Empowers non-technical stakeholders to explore predictive analytics.

Scalability: Can be reused across departments for different datasets and use cases (e.g. churn prediction, sales forecasting, credit scoring).

Conclusion

The Auto ML Prediction App provides an efficient and user-friendly solution for applying machine learning to business data. By automating key steps—such as data cleaning, model selection, training, and evaluation—the app makes predictive analytics accessible to a wider range of users, including those without technical expertise.

With just a few clicks, decision-makers can uncover patterns in their data, forecast future outcomes, and support strategic initiatives with data-driven insights. Whether used for classification tasks like customer churn prediction or regression use cases such as sales forecasting, this tool streamlines the workflow from raw data to actionable intelligence.

As organizations increasingly adopt data-driven strategies, tools like this empower teams to innovate faster, reduce dependency on technical resources, and make smarter decisions with confidence.
