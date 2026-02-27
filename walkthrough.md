# ✈️ SkyGuard: Flight Delay Prediction System Walkthrough

This document provide a technical deep-dive into the flight delay prediction pipeline.

## 📋 5-Step Analysis Pipeline

### 1. Data Collection
- **Source**: NYC Flights 2013 + NOAA Weather.
- **Granularity**: Merged hourly weather data with individual flight records (330k+).
- **Enrichment**: Integrated OpenFlights database for descriptive airport and country names.

### 2. Time-Based EDA
- **Hourly Peak**: Delays grow cumulatively throughout the day, peaking at **>300 minutes** after 10 PM.
- **Monthly Trend**: Summer months (June/July) show the highest operational stress.

### 3. Feature Engineering
- **Temporal**: Month, Day, Hour (Diurnal cycle).
- **Environment**: Temperature, Humidity, Precipitation, Pressure, Visibility.
- **Logistics**: Origin, Destination, Carrier performance.

### 4. XGBoost Modeling
- **Algorithm**: Gradient Boosting (XGBoost Classifier).
- **Accuracy**: **82.17%**
- **ROC-AUC**: **0.8141**

### 5. Factor Analysis (Why Delays Happen)
1. **Departure Hour (33.6%)**: The single most critical predictor.
2. **Humidity (9.6%)**: Key weather disruptor.
3. **Carrier (7.8%)**: Operational efficiency variance.

---

## 🎨 Professional Dashboard
- **Location Intelligence**: Displays full names like "Newark Liberty International (United States)".
- **Dynamic Predictions**: Probability dial showing the model's confidence in real-time.
- **Outcome Messaging**: Clear Red/Green status for delayed vs. on-time flights.

### 🚀 Usage
- Full Analysis: `python analysis_pipeline.py`
- Web Portal: `python app.py`
