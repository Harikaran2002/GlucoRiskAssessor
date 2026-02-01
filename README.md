# GlucoRiskAssessor

a. Problem Statement

The aim of this project is to develop a machine learning–based diabetes prediction system that can analyze patient clinical data to identify early-stage diabetes. The system allows users to upload test datasets, select from multiple trained classification models, and view prediction results along with comprehensive evaluation metrics and visual performance analysis.

b. Dataset Description

This dataset comprises crucial sign and symptom data of individuals who either exhibit early signs of diabetes or are at risk of developing diabetes. The variables included in the dataset provide valuable insights into potential indicators of diabetes onset. The dataset encompasses diverse information, ranging from demographic details to specific symptoms associated with diabetes.

The dataset consists of 16 input attributes and one target variable, each representing a clinical sign, symptom, or demographic factor relevant to diabetes diagnosis.

Attribute Description

    Age: Represents the age of the individual in years. Age is a significant risk factor for diabetes, with prevalence increasing in older populations.

    Gender: Indicates the biological sex of the individual (Male or Female), which may influence susceptibility and symptom presentation.

    Polyuria: Indicates frequent urination (1 = Yes, 0 = No), a common early symptom of diabetes.

    Polydipsia: Represents excessive thirst (1 = Yes, 0 = No), often associated with elevated blood glucose levels.

    Sudden Weight Loss: Indicates unexplained or rapid loss of body weight (1 = Yes, 0 = No), a potential warning sign of diabetes.

    Weakness: Denotes general fatigue or loss of strength (1 = Yes, 0 = No), commonly reported by diabetic individuals.

    Polyphagia: Represents excessive hunger (1 = Yes, 0 = No), which can occur due to ineffective glucose utilization.

    Genital Thrush: Indicates the presence of fungal infections (1 = Yes, 0 = No), often linked to high blood sugar levels.

    Visual Blurring: Indicates blurred vision (1 = Yes, 0 = No), caused by fluctuations in blood glucose affecting eye lenses.

    Itching: Represents persistent itching sensations (1 = Yes, 0 = No), which may be associated with skin infections or dehydration.

    Irritability: Captures mood changes or irritability (1 = Yes, 0 = No), sometimes observed in individuals with glucose imbalance.

    Delayed Healing: Indicates slow wound or injury healing (1 = Yes, 0 = No), a characteristic symptom of diabetes due to poor circulation.

    Partial Paresis: Represents partial muscle weakness or paralysis (1 = Yes, 0 = No), which may occur in advanced or unmanaged diabetes cases.

    Muscle Stiffness: Indicates stiffness or reduced flexibility in muscles (1 = Yes, 0 = No), possibly related to metabolic imbalances.

    Alopecia: Represents hair loss (1 = Yes, 0 = No), which may be linked to hormonal or metabolic disorders.

    Obesity: Indicates whether the individual is obese (1 = Yes, 0 = No), a major risk factor for the development of type 2 diabetes.

    Class (Target Variable): Represents the diabetes status of the individual, where 1 indicates diabetic and 0 indicates non-diabetic.

All symptom-based attributes are encoded as binary variables to simplify model interpretation and facilitate efficient training of machine learning classifiers.

c. Machine Learning Models Used
    The following six supervised machine learning classification models were implemented and evaluated:
        Logistic Regression
        Decision Tree Classifier
        K-Nearest Neighbors (KNN)
        Naive Bayes Classifier
        Random Forest Classifier
        XGBoost (Extreme Gradient Boosting)

Evaluation Metrics

    The models were evaluated using the following performance metrics:
        Accuracy
        Area Under the ROC Curve (AUC)
        Precision
        Recall
        F1-Score
        Matthews Correlation Coefficient (MCC)

Model Performance Comparison
S.No	ML Model	Accuracy	AUC	Precision	Recall	F1	MCC
1	Logistic Regression	0.9067	0.9575	0.9744	0.8636	0.9157	0.8194
2	Decision Tree	0.9467	0.9666	1.0000	0.9091	0.9524	0.8973
3	KNN	1.0000	1.0000	1.0000	1.0000	1.0000	1.0000
4	Naive Bayes	0.8533	0.9194	0.9024	0.8409	0.8706	0.7042
5	Random Forest	0.9733	1.0000	1.0000	0.9545	0.9767	0.9469
6	XGBoost	0.9867	1.0000	1.0000	0.9773	0.9885	0.9730

Observations
S.No	ML Model	Observation
1	Logistic Regression	Achieved strong discriminative performance with high precision but comparatively lower recall, indicating conservative predictions.
2	Decision Tree	Demonstrated reliable classification with perfect precision and strong MCC, though susceptible to overfitting.
3	KNN	Achieved perfect scores across all metrics, indicating excellent performance, though potentially sensitive to dataset characteristics.
4	Naive Bayes	Recorded the lowest overall performance due to its strong independence assumptions.
5	Random Forest	Showed robust generalization with high accuracy, perfect precision, and strong MCC.
6	XGBoost	Emerged as the best-performing model with excellent balanced performance across all evaluation metrics.

Conclusion
    Among the evaluated models, XGBoost and Random Forest demonstrated superior and consistent performance across all metrics, making them the most suitable models for early-stage diabetes prediction. Although KNN achieved perfect scores, its performance may be sensitive to dataset size and feature scaling. Naive Bayes showed comparatively lower performance, highlighting the limitations of probabilistic assumptions in this context.