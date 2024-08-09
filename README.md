# PCA-for-Diagnosis-of-Heart-Disease

## Overview

In this project, Principal Component Analysis (PCA) is applied to study the performance of clustering algorithms on the Cleveland heart disease dataset. The report is divided into two parts.
- In the first section, we apply PCA on a dataset of 303 people corresponding to the 13 specific features. We found that 55% of the explained variance was in the first 2 principal components.
- In the second section, we have used logistic regression for classifying the observations.

## Dataset
The dataset contains medical records of 303 patients who had heart failure, collected during their follow-up period, where each patient profile has 14 clinical features. [Dataset](./data/Cleveland_data.csv) is under data secion of the repository.

### Attributes Information:
|   Attribute    |  Type  |	       Description        |
| -------------- | ------ | ------------------------- |
| Age	         |   int  | Age of a patient [in years]
| Sex	         |   int  | Gender of the patient [1=Male, 0=Female]
| ChestPain (Cp) |   int  | Chest pain type [1=Typical Angina, 2=Atypical Angina, 3=Non-Anginal Pain, 4=Asymptomatic]
| TrestBPS	     |   int  | Resting blood pressure [in mm Hg] (Normal blood pressure - 120/80 Hg)
| Chol      	 |   int  | Serum cholestrol level in blood [in mg/dl] (Normal cholesterol level below for adults 200mg/dL)
| FBS   	     |   int  | Fasting Blood Sugar [1=true, 0=false] (Normal less than 100mg/dL for non diabetes for diabetes 100-125mg/dL)
| RestECG	     |   int  | Resting electrocardiogram results [0=Normal, 1=having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 2=showing probable or definite left ventricular hypertrophy by Estes' criteria]
| Thalach	     |   int  | Maximum heart rate achieved [between 60 and 202]
| Exang          |   int  | Exercise-induced angina [1=Yes, 0=No]
| Oldpeak	     |   int  | ST depression induced by exercise relative to rest
| Slope	         |   int  | The slope of the peak exercise ST segment [1=upsloping, 2=flat, 3=downsloping]
| Ca        	 |   int  | Number of major vessels (0–3) colored by fluoroscopy
| Thal           |   int  | 3 = normal; 6 = fixed defect; 7 = reversible defect

## Steps Involved
This is a summary of the procedures involved in applying machine learning algorithms to Principal Component Analysis (PCA) for a heart disease prediction project.

***1. Data Collection and Preprocessing***

Gather a dataset containing relevant features related to heart health (e.g., age, blood pressure, cholesterol levels, etc.). Handle missing values, encode categorical variables, and normalize/standardize the data.

***2. Exploratory Data Analysis (EDA)***

Perform descriptive statistics, visualizations, and correlation analysis to understand the dataset. Assess feature importance and relationships to gain insights.

***3. Feature Selection and PCA***

Identify features relevant for predicting heart disease. Apply PCA to reduce the dimensionality of the dataset while retaining important information. Determine the number of principal components to keep (using variance explained or scree plot).

***4. Split Data for Training and Testing***

Divide the dataset into training and testing sets (e.g., 70-30 or 80-20 split).

***5. Model Selection and Training***

Choose appropriate machine learning algorithms (e.g., Logistic Regression, Random Forest, SVM) for classification. Fit the model on the training data.

***6. Model Evaluation***

Evaluate the model's performance using the testing data (accuracy, precision, recall, F1-score, ROC curve, etc.). Use cross-validation to assess model robustness.

***7. Tuning and Optimization***

Fine-tune hyperparameters of the models to improve performance (e.g., GridSearchCV or RandomizedSearchCV).

***8. Prediction and Interpretation***

Make predictions on new/unseen data using the trained model. Interpret the results and assess the factors contributing to heart disease prediction.

## Doc
Contains the [report](./doc/project_report.pdf) for in-depth understanding of the project.