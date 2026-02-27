import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import os

def load_airport_mapping():
    mapping = {}
    if os.path.exists('airports.dat'):
        df = pd.read_csv('airports.dat', header=None)
        for _, row in df.iterrows():
            iata = str(row[4]).strip('"')
            if iata and iata != '\\N':
                mapping[iata] = f"{str(row[1]).strip('\"')} ({str(row[3]).strip('\"')})"
    return mapping

def run_pipeline():
    airport_map = load_airport_mapping()
    
    # 1. Collect flight schedule and delay data
    print("Step 1: Collecting flight schedule and delay data...")
    if not os.path.exists('flights.csv') or not os.path.exists('weather.csv'):
        print("Data files missing. Please ensure flights.csv and weather.csv are in the directory.")
        return
    
    flights = pd.read_csv('flights.csv')
    weather = pd.read_csv('weather.csv')
    
    # Cleaning
    if flights.columns[0] == '' or 'Unnamed' in flights.columns[0]: flights = flights.iloc[:, 1:]
    if weather.columns[0] == '' or 'Unnamed' in weather.columns[0]: weather = weather.iloc[:, 1:]

    # 2. Perform time-based exploratory analysis
    print("\nStep 2: Performing time-based exploratory analysis...")
    flights['is_delayed'] = (flights['dep_delay'] > 15).astype(int)
    
    # Monthly trend
    monthly_delays = flights.groupby('month')['dep_delay'].mean()
    print("\n--- Average Delay by Month ---")
    print(monthly_delays)
    
    # Origins with descriptive names
    print("\n--- Top Delay-Prone Origins (Descriptive) ---")
    origin_delays = flights.groupby('origin')['dep_delay'].mean().sort_values(ascending=False).head(5)
    for code, delay in origin_delays.items():
        print(f"{airport_map.get(code, code)}: {delay:.2f} mins")
    hourly_delays = flights.groupby('hour')['dep_delay'].mean()
    print("\nAverage Delay by Hour (Top 5):\n", hourly_delays.sort_values(ascending=False).head(5))

    # 3. Engineer features like departure time, season, and weather conditions
    print("\nStep 3: Engineering features...")
    # Merging flight and weather data
    data = pd.merge(flights, weather, on=['year', 'month', 'day', 'hour', 'origin'], how='inner')
    
    # Feature Selection
    features = ['month', 'day', 'hour', 'origin', 'dest', 'carrier', 
                'temp', 'dewp', 'humid', 'wind_speed', 'precip', 'pressure', 'visib']
    
    X = data[features].copy()
    y = data['is_delayed']
    
    # Encoding categoricals
    le = LabelEncoder()
    for col in ['origin', 'dest', 'carrier']:
        X[col] = le.fit_transform(X[col].astype(str))
        
    # Handling missing values
    X = X.fillna(X.mean())

    # 4. Train classification or regression models
    print("\nStep 4: Training classification model (XGBoost)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {model.score(X_test, y_test):.4f}")
    print(f"Model ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")

    # 5. Analyze important factors causing delays
    print("\nStep 5: Analyzing important factors causing delays...")
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("Top Influential Factors:\n", importances)
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    run_pipeline()
