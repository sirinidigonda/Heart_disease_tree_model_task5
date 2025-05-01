Heart Disease Prediction - Decision Trees and Random Forests
Objective
This project aims to apply decision tree and random forest algorithms for classification and prediction of heart disease. The models are trained on the Heart Disease dataset and evaluated using accuracy, cross-validation, and feature importance.

Tools Used
Python: Programming language used for implementation.
Scikit-learn: For implementing Decision Trees, Random Forest, and model evaluation.
Pandas: For data handling and manipulation.
Matplotlib and Graphviz: For visualizing the decision tree.
NumPy: For numerical operations.
Cross-Validation: To evaluate model performance and generalize results.

Dataset
The dataset used is the Heart Disease dataset which contains various health attributes for individuals, such as age, cholesterol, and exercise habits. The target variable is target, which indicates whether the individual has heart disease (1) or not (0).

Steps
Data Preprocessing:
The dataset was loaded and inspected for any missing values or discrepancies.
No missing values were found.
Summary statistics of the dataset were generated to understand its distribution.

Model Training:
A Decision Tree Classifier was trained on the dataset.
A Random Forest Classifier was trained on the same dataset for comparison.

Model Evaluation:
The performance of the models was evaluated using accuracy.
The Random Forest model achieved a slightly higher accuracy compared to the Decision Tree model.

Cross-Validation:
The Random Forest model was evaluated using cross-validation, which gave a mean accuracy of 99.71%.

Feature Importance:
The Random Forest model was used to interpret the importance of various features in predicting heart disease. This helps in understanding which health attributes play a significant role in predicting the target.

Results:
Decision Tree Accuracy: 97.08%
Random Forest Accuracy: 98.05%
Cross-Validation Scores: [1.0, 1.0, 1.0, 1.0, 0.985]
Mean Cross-Validation Accuracy: 99.71%

Files in the Repository:
heart_disease_tree_model.py: Python script for loading the dataset, training the models, and evaluating their performance.
heart_disease_data.csv: The dataset used in the analysis.
decision_tree_visualization.png: Visualization of the trained decision tree.

Conclusion:
The Random Forest model performs slightly better than the Decision Tree model. Cross-validation shows that the model is reliable and consistent. Feature importance analysis reveals key factors that affect the prediction of heart disease.
