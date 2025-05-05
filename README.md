# MRP-Group-19

# Employee Salary Prediction

Employee salary prediction involves determining an employee's salary level based on various personal and organizational factors. This analysis helps organizations understand the factors influencing salary levels and enables better decision-making.


## 📚 Table of Contents

1. Use Cases for Salary Prediction
2. Project Description
3. Tools & Technologies
4. Dataset Overview
5. How It Works
6. Installation / Setup
7. Import Required Libraries
8. Usage
9. Features
10. ROC Curve Example
11. Power BI Integration
12. Dashboard Highlights
13. Contributing
14. License

## 🔍 Use Cases for Salary Prediction

* **Compensation Benchmarking**: Compare salaries across departments and roles to ensure fairness.
* **Budget Forecasting**: Support better financial planning by analyzing salary distributions.
* **Career Path Analysis**: Identify salary growth trends tied to performance and promotions.

---

## 📊 Project Description

Our project aims to predict employee salaries using machine learning techniques. The model is trained on an HR dataset and treats salary as an ordinal variable. The solution also includes:

* Python scripts for data preprocessing and prediction.
* A graphical user interface (GUI) for user input.
* A Power BI dashboard for data visualization.

---

## 🛠️ Tools & Technologies

* **Programming**: Python (Pandas, Scikit-learn, Joblib)
* **Machine Learning Model**: Random Forest Classifier
* **Visualization**: Power BI & Figma
* **Preprocessing**: Label Encoding, Standardization

---

## 📁 Dataset Overview

* **Input**: `HR_dataset.csv`

  * Features:

    * Satisfaction Level
    * Last Evaluation
    * Number of Projects
    * Average Monthly Hours
    * Department
    * Promotion in the Last 5 Years
    * Salary (categorical target)

* **Output**:

  * `updated_salary_predictions.csv`
  * `HR_sample_profile_updated.csv`
  * Contains a new column: `predicted_salary`

---

## 🚀 How It Works

1. **Data Preprocessing**

   * Handle missing values
   * Encode categorical columns
   * Scale numerical data

2. **Model Training**

   * Random Forest Classifier
   * Evaluate performance using accuracy and classification metrics

3. **Salary Prediction**

   * Predict salary categories for all employees
   * Add `predicted_salary` to the dataset

4. **Export and Visualize**

   * Save updated dataset as `.csv`
   * Import into Power BI for interactive analysis

The core Python code used in this project is provided in the following notebook:

▶️ **[Project-Predection of employee salaries.py-.ipynb]( https://github.com/Mahitha1456/MRP-Group-19/blob/main/Prediction%20of%20salaries%20Python%20code.pbix)

---

## ⚙️ Installation / Setup

```bash
git clone <repository-url>
cd employee-salary-prediction
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📚 Import Required Libraries

```python
pip install joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load data
df = pd.read_csv('HR_dataset.csv')
df.head()
```

---

## ⚡ Usage

To run the prediction pipeline:

```bash
python generate_updated_dataset.py
```

To use the GUI:

* Launch the GUI file (if implemented)
* Input employee details
* View predicted salary category

---

## ✨ Features

* Predict employee salary levels
* GUI for interaction (optional)
* Dynamic Power BI dashboard
* Classification based on multiple HR attributes

---

## 🔍 ROC Curve Example

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

logit_roc_auc = roc_auc_score(y_test, model_logistic.predict_proba(X_test), multi_class='ovr')
fpr, tpr, _ = roc_curve(y_test, model_logistic.predict_proba(X_test)[:, 1], pos_label=1)

rf_roc_auc = roc_auc_score(y_test, model_rf.predict_proba(X_test), multi_class='ovr')
rf_fpr, rf_tpr, _ = roc_curve(y_test, model_rf.predict_proba(X_test)[:, 1], pos_label=1)

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
```

---

## 📈 Power BI Integration

Power BI plays a crucial role in our project by turning the predictions and employee data into actionable insights. The updated dataset generated from our Python code was loaded into Power BI to build an interactive and dynamic dashboard. Key features of the Power BI integration include:

* **Dynamic Filtering**: Users can filter results by promotion status, satisfaction level, predicted salary, and department.
* **Real-Time Analysis**: HR managers can interact with visuals to instantly explore trends in salary, promotion, and employee workload.
* **Visual Representation**: Graphical analysis helps in understanding workforce patterns and making decisions on compensation and career planning.

The Power BI file used is:

* 🔗 [Prediction of salaries Python code.pbix](https://github.com/Mahitha1456/MRP-Group-19/blob/main/Prediction%20of%20salaries%20Python%20code.pbix)

---

## 📊 Dashboard Highlights

* **Filters**:

  * Promotion Status
  * Predicted Salary Range
  * Satisfaction Level

* **Visuals**:

  * Pie Chart: Promotions by Salary
  * Bar Chart: Satisfaction by Salary
  * Scatter Plot: Monthly Hours vs Salary
  * Donut Chart: Turnover by Department

* **Power BI File**: [Prediction of salaries Python code.pbix](./Prediction%20of%20salaries%20Python%20code.pbix)

* **Figma Prototype**: [Figma Dashboard](https://www.figma.com/design/Buw5I1UcFNjnnVjvszxyk1/Untitled?node-id=0-1&p=f&t=musI23SXBkoxHrsf-0)

* **Figma File**: [Employee\_Salary\_Dashboard.fig](./Employee_Salary_Dashboard.fig)

---

## 💡 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss your proposed changes.

---

## 📄 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software with proper attribution.

 








