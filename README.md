# TelcoPulse AI: End-to-End Customer Churn Classification & Streamlit Diagnostic Suite

**Course**: M.Tech (AIML) — Machine Learning (Assignment 2)  
**Institution**: BITS Pilani (Work Integrated Learning Programmes Division)  
**Student Name**: Satyaki Pal  
**BITS ID**: 2026AJ05808  

---

## a. Problem Statement
Customer attrition (churn) is a primary revenue risk in the telecommunications industry, where acquiring a new subscriber costs five to seven times more than retaining an existing customer. Telecom providers capture rich subscriber telemetry across contract terms, account tenure, internet connectivity tiers, security add-ons, and billing patterns.

The objective of this project is to design, evaluate, and deploy an end-to-end supervised Machine Learning classification pipeline that predicts whether a customer will churn (`Churn = 1`) or remain retained (`Churn = 0`). Five distinct classification models—**Logistic Regression**, **Decision Tree Classifier**, **K-Nearest Neighbor (KNN) Classifier**, **Gaussian Naive Bayes**, and **Support Vector Machine (SVM)**—are implemented on the same preprocessed dataset, evaluated across six standardized metrics (**Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient**), and served via an interactive web application on **Streamlit Community Cloud** that evaluates uploaded holdout test data without retraining.

---

## b. Dataset Description
* **Dataset Name**: Telco Customer Churn Dataset
* **Public Repository**: Kaggle (IBM Business Analytics Public Benchmark)
* **Dataset URL**: [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Classification Problem Type**: Binary Classification (`0` = Retained / No Churn, `1` = Churned)
* **Instance Size**: **7,043 total instances** (Exceeds the minimum requirement of 500 instances)
  * **Training Split (80%)**: 5,634 instances (stratified by target class; used to fit `StandardScaler` and train all 5 models)
  * **Holdout Test Split (20%)**: 1,409 instances exported as `test_data.csv` (1,035 Class `0`, 374 Class `1`) for live Streamlit evaluation
* **Feature Size**: **19 input features** + 1 target column `Churn` (Exceeds the minimum requirement of 12 features):

| # | Feature Name | Description | Data Type / Encoding |
| :---: | :--- | :--- | :--- |
| 1 | `gender` | Customer gender | Binary (`0`: Female, `1`: Male) |
| 2 | `SeniorCitizen` | Whether the customer is a senior citizen (65+ years) | Binary (`0`: No, `1`: Yes) |
| 3 | `Partner` | Whether the customer has a partner | Binary (`0`: No, `1`: Yes) |
| 4 | `Dependents` | Whether the customer has dependents | Binary (`0`: No, `1`: Yes) |
| 5 | `tenure` | Number of months the customer has stayed with the company | Numerical (`0` to `72` months) |
| 6 | `PhoneService` | Whether the customer has phone service | Binary (`0`: No, `1`: Yes) |
| 7 | `MultipleLines` | Whether the customer has multiple phone lines | Ordinal (`0`: No phone service, `1`: No, `2`: Yes) |
| 8 | `InternetService` | Customer's internet service provider tier | Ordinal (`0`: No, `1`: DSL, `2`: Fiber optic) |
| 9 | `OnlineSecurity` | Whether the customer has online security add-on | Ordinal (`0`: No internet, `1`: No, `2`: Yes) |
| 10 | `OnlineBackup` | Whether the customer has online backup add-on | Ordinal (`0`: No internet, `1`: No, `2`: Yes) |
| 11 | `DeviceProtection` | Whether the customer has device protection add-on | Ordinal (`0`: No internet, `1`: No, `2`: Yes) |
| 12 | `TechSupport` | Whether the customer has tech support add-on | Ordinal (`0`: No internet, `1`: No, `2`: Yes) |
| 13 | `StreamingTV` | Whether the customer has streaming TV add-on | Ordinal (`0`: No internet, `1`: No, `2`: Yes) |
| 14 | `StreamingMovies` | Whether the customer has streaming movies add-on | Ordinal (`0`: No internet, `1`: No, `2`: Yes) |
| 15 | `Contract` | The contract term of the customer | Ordinal (`0`: Month-to-month, `1`: One year, `2`: Two year) |
| 16 | `PaperlessBilling` | Whether the customer has paperless billing enabled | Binary (`0`: No, `1`: Yes) |
| 17 | `PaymentMethod` | The customer's payment method | Categorical (`0`: Electronic check, `1`: Mailed check, `2`: Bank transfer, `3`: Credit card) |
| 18 | `MonthlyCharges` | The amount charged to the customer monthly (`USD`) | Continuous (`$18.25` to `$118.75`) |
| 19 | `TotalCharges` | The total cumulative amount charged to the customer (`USD`) | Continuous (`$0.00` to `$8,684.80`) |
| **Target** | `Churn` | Whether the customer churned | Binary (`0` = Retained, `1` = Churned) |

---

## c. Github Repository Link
* **GitHub Repository Link**: [https://github.com/satyakipal/telco-churn-ml-suite](https://github.com/satyakipal/telco-churn-ml-suite)
* **Live Streamlit App Link**: [https://2026aj05808-bits-aiml-ml-assignment2.streamlit.app](https://2026aj05808-bits-aiml-ml-assignment2.streamlit.app)

### Repository Structure
* `app.py` — Interactive Streamlit web application
* `train_models.py` — Model training, evaluation & artifact serialization script
* `requirements.txt` — Python dependencies for Streamlit Community Cloud
* `README.md` — Complete assignment documentation & evaluation tables
* `test_data.csv` — Holdout test data (1,409 rows: 19 features + `Churn` target)
* `model/` — Saved trained models & preprocessing artifacts (`scaler.joblib`, `metadata.json`, `logistic_regression.joblib`, `decision_tree.joblib`, `knn_classifier.joblib`, `naive_bayes.joblib`, `svm_classifier.joblib`)

---

## d. Models used

### 1. Comparison Table of Evaluation Metrics (Holdout Test Set, N = 1,409)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7984 | **0.8469** | 0.6372 | 0.5588 | 0.5954 | 0.4638 |
| **Decision Tree** | 0.7828 | 0.8291 | 0.6156 | 0.4840 | 0.5419 | 0.4073 |
| **KNN** | 0.7715 | 0.8281 | 0.5707 | 0.5615 | 0.5660 | 0.4110 |
| **Naive Bayes** | 0.7289 | 0.8262 | 0.4934 | **0.7968** | **0.6094** | 0.4471 |
| **Support Vector Machine** | **0.8077** | 0.8066 | **0.7004** | 0.4813 | 0.5705 | **0.4653** |

---

### 2. Observations on the Performance of Each Model on the Chosen Dataset

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves the highest **AUC Score (0.8469)** and second-highest **Accuracy (0.7984)** and **MCC (0.4638)**. Because key churn drivers (`tenure`, `Contract` term, and `MonthlyCharges`) exhibit strong monotonic relationships with log-odds of attrition, L2-regularized Logistic Regression produces well-calibrated probability estimates and strong rank-ordering across all decision thresholds. |
| **Decision Tree** | Delivers an interpretable baseline with **Accuracy of 0.7828**, **AUC of 0.8291**, and **Precision of 0.6156**. Constraining tree depth (`max_depth=6`, `min_samples_leaf=10`) prevents overfitting and captures primary hierarchical rules (such as month-to-month fiber optic subscribers), though discrete axis-aligned splits yield lower **Recall (0.4840)** and **MCC (0.4073)** than smooth margin classifiers. |
| **KNN** | Records balanced mid-tier performance (**Accuracy: 0.7715**, **AUC: 0.8281**, **Precision: 0.5707**, **Recall: 0.5615**, **F1: 0.5660**). While `StandardScaler` normalization ensures equal feature weighting, Euclidean distance across 19 mixed continuous and ordinal dimensions experiences mild curse-of-dimensionality dilution compared to parametric models. |
| **Naive Bayes** | Achieves the highest **Recall (0.7968)** and highest **F1 Score (0.6094)** across all five classifiers, successfully capturing nearly 80% of all true churning subscribers. However, because Gaussian Naive Bayes assumes conditional independence among correlated billing and service features (`tenure`, `TotalCharges`, and streaming/security add-ons), it over-predicts the positive class, resulting in lower **Precision (0.4934)** and **Accuracy (0.7289)**. |
| **Support Vector Machine** | Achieves the highest **Accuracy (0.8077)**, highest **Precision (0.7004)**, and highest **MCC (0.4653)** using an RBF kernel (`C=1.2`). The non-linear kernel constructs a strict maximum-margin boundary in transformed feature space that minimizes False Positives, though at the default 0.50 threshold it operates conservatively on **Recall (0.4813)** with an **AUC of 0.8066**. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)** and **Logistic Regression** emerge as the top winners depending on the operational objective: **Support Vector Machine** is the overall winner for discrete classification accuracy (**0.8077**), precision (**0.7004**), and Matthews Correlation Coefficient (**0.4653**), whereas **Logistic Regression** is the winner for probability ranking and threshold flexibility with the highest **ROC-AUC (0.8469)**. *(For retention campaigns prioritizing maximum churner capture, **Naive Bayes** leads in **Recall at 0.7968** and **F1 at 0.6094**).* |
