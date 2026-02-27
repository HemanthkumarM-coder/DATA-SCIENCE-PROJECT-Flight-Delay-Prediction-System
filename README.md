# ✈️ SkyGuard: Flight Delay Prediction System

A professional machine learning application that predicts flight departure delays based on historic aviation data and real-time weather conditions.

## 📊 Features
- **Statistical Analysis**: Deep-dive into NYC 2013 flight data (330k+ records).
- **Meteorological Integration**: Synchronized with NOAA weather data (Temperature, Visibility, Wind Speed).
- **Location Intelligence**: Descriptive airport names and country labels using the OpenFlights database.
- **Predictive Engine**: High-performance XGBoost model with ~82% accuracy.
- **Premium Dashboard**: Responsive dark-mode UI for intuitive delay estimation.

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the 5-Step Analysis
Execute the full data science pipeline:
```bash
python analysis_pipeline.py
```

### 3. Start the Web Dashboard
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

## 📋 Documentation
- **`walkthrough.md`**: Detailed technical breakdown of the architecture, EDA, and model metrics.
- **`eda_summary.txt`**: Statistical summary of initial findings.
