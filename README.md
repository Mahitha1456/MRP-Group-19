# MRP-Group-19
# Employee Salary Prediction

Employee salary prediction involves determining an employee's salary level based on various personal and organizational factors. This analysis helps organizations understand the factors influencing salary levels and enables better decision-making.

Use Cases for Salary Prediction:

Compensation Benchmarking: Organizations can compare salaries across various departments and positions to ensure fairness and competitiveness.
Budget Forecasting: Helps organizations allocate funds more effectively by understanding the distribution of salaries.
Career Path Analysis: Identifies the potential salary growth associated with different career paths, assisting in strategic planning.
In this analysis, the goal is to predict the salary level of an employee based on features like performance evaluation scores, work experience, department, and past promotion.

**Table of Contents:**
1. Project Description
2. Import Library dataset
3. Usage
4. Features
5. Dashboard
6. Contributing
7. License

**Project Description** Our project aims to predict employee salaries using machine learning techniques. The model is trained on an HR dataset and treats salary as an ordinal variable. The project also shows a graphical user interface (GUI) for user interaction and a dashboard for data visualization.

**Installation / Setup**

Clone the repository:
git clone https://github.com/yourusername/employee-salary-prediction.git
cd employee-salary-prediction
Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies:
pip install -r requirements.txt

**Import Library dataset** 
pip install joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv('dataset.csv')
df.head()

**Usage**
Run the salary prediction script:
python main.py

Use the GUI to input employee details and view the predicted salary category.

**Project structure**
**Features**
Predicts employee salaries based on multiple attributes.
GUI for easy interaction.
Data visualization dashboard.
Supports salary classification as an ordinal variable.

**ROC Curve**
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
# Logistic Regression
logit_roc_auc = roc_auc_score(y_test, model_logistic.predict_proba(X_test), multi_class='ovr')
fpr, tpr, thresholds = roc_curve(y_test, model_logistic.predict_proba(X_test)[:, 1], pos_label=1)
# Random Forest
rf_roc_auc = roc_auc_score(y_test, model_rf.predict_proba(X_test), multi_class='ovr')
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, model_rf.predict_proba(X_test)[:, 1], pos_label=1)

# Plot ROC Curves
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

![image](https://github.com/user-attachments/assets/067eedcf-df2a-48bc-b104-a832b5974571)


**Dashboard**
This is our interactive dashboard: 
https://www.figma.com/design/Buw5I1UcFNjnnVjvszxyk1/Untitled?node-id=0-1&p=f&t=musI23SXBkoxHrsf-0

[image](https://github.com/user-attachments/assets/c1c8283e-4d3a-4607-b42e-0a28654ed36d)

**Contributing**
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to modify.

**License**
This project is licensed under the MIT License. You are free to use, modify, and distribute this software with proper attribution. See the LICENSE file for details.


