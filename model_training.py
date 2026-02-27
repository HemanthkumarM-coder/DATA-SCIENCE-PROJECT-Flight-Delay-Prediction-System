import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load processed/raw data
print("Loading datasets for modeling...")
flights = pd.read_csv('flights.csv')
weather = pd.read_csv('weather.csv')

# Cleaning index columns
if flights.columns[0] == 'unnamed: 0' or flights.columns[0] == '':
    flights = flights.iloc[:, 1:]
if weather.columns[0] == 'unnamed: 0' or weather.columns[0] == '':
    weather = weather.iloc[:, 1:]

# Target variable: 1 if delayed > 15 mins, else 0
flights['is_delayed'] = (flights['dep_delay'] > 15).astype(int)

# Merge
print("Merging data...")
data = pd.merge(flights, weather, on=['year', 'month', 'day', 'hour', 'origin'], how='inner')

# Feature Selection
features = ['month', 'day', 'hour', 'origin', 'dest', 'carrier', 
            'temp', 'dewp', 'humid', 'wind_speed', 'precip', 'pressure', 'visib']
target = 'is_delayed'

X = data[features].copy()
y = data[target]

# Label Encoding categorical features
print("Encoding categorical variables...")
le_origin = LabelEncoder()
le_dest = LabelEncoder()
le_carrier = LabelEncoder()

X['origin'] = le_origin.fit_transform(X['origin'])
X['dest'] = le_dest.fit_transform(X['dest'])
X['carrier'] = le_carrier.fit_transform(X['carrier'])

# Handling missing values in weather (simple fill)
X = X.fillna(X.mean())

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeling
print("Training XGBoost Classifier...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Saving Model and Encoders
print("\nSaving model and artifacts...")
joblib.dump(model, 'flight_delay_model.joblib')
joblib.dump(le_origin, 'le_origin.joblib')
joblib.dump(le_dest, 'le_dest.joblib')
joblib.dump(le_carrier, 'le_carrier.joblib')

print("Final list of Carriers:", le_carrier.classes_)
print("Final list of Origins:", le_origin.classes_)

print("\nModel and artifacts saved successfully.")
