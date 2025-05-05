# updated_dataset.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Step 1: Load the dataset
df = pd.read_csv('HR_dataset.csv')  

# Step 2: Handle missing values (if any)
df = df.dropna()

# Step 3: Encode categorical columns
label_encoders = {}
categorical_cols = ['department', 'salary']  # assuming 'salary' is the target class (low, medium, high)

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 4: Define features and target
X = df.drop(['salary'], axis=1)  
y = df['salary']

# Step 5: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Train a classifier (Random Forest in this case)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Predict salary category for entire dataset
predicted_salary_encoded = model.predict(X_scaled)

# Step 9: Decode the predicted salary back to original labels
salary_le = label_encoders['salary']
df['predicted_salary'] = salary_le.inverse_transform(predicted_salary_encoded)

# Step 10: Save the updated dataset to CSV for Power BI
df.to_csv('updated_hr_dataset_with_predictions.csv', index=False)
print("Updated dataset saved as 'updated_hr_dataset_with_predictions.csv'.")

# Step 11: Optionally save model and scaler
joblib.dump(model, 'salary_prediction_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
