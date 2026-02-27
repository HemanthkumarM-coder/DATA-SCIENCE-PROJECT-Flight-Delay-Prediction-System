import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
print("Loading datasets...")
flights = pd.read_csv('flights.csv')
weather = pd.read_csv('weather.csv')

# Remove the first index column if it exists (empty name)
if flights.columns[0] == 'unnamed: 0' or flights.columns[0] == '':
    flights = flights.iloc[:, 1:]
if weather.columns[0] == 'unnamed: 0' or weather.columns[0] == '':
    weather = weather.iloc[:, 1:]

# Basic cleaning
flights['dep_delay'] = flights['dep_delay'].fillna(0)
flights['arr_delay'] = flights['arr_delay'].fillna(0)

# 1. Time-based Exploration: Average Delay by Month
print("Analyzing delays by month...")
monthly_delay = flights.groupby('month')['dep_delay'].mean()

# 2. Hourly Trends: Average Delay by Hour
print("Analyzing delays by hour...")
hourly_delay = flights.groupby('hour')['dep_delay'].mean()

# 3. Weather Correlation
# Merge weather with flights
print("Merging flights and weather...")
# Note: Weather is hourly. Flights also have 'hour'.
merged_df = pd.merge(flights, weather, on=['year', 'month', 'day', 'hour', 'origin'], how='inner')

# Correlation with visibility and precipitation
weather_corr = merged_df[['dep_delay', 'temp', 'precip', 'visib', 'wind_speed']].corr()['dep_delay']

# Summary Output
with open('eda_summary.txt', 'w') as f:
    f.write("EDA Summary - Flight Delay Prediction\n")
    f.write("=====================================\n\n")
    f.write(f"Total flights analyzed: {len(flights)}\n")
    f.write(f"Average Departure Delay: {flights['dep_delay'].mean():.2f} mins\n\n")
    
    f.write("Monthly Average Delays:\n")
    f.write(monthly_delay.to_string() + "\n\n")
    
    f.write("Hourly Average Delays:\n")
    f.write(hourly_delay.to_string() + "\n\n")
    
    f.write("Weather Correlations with Departure Delay:\n")
    f.write(weather_corr.to_string() + "\n")

print("EDA Summary saved to eda_summary.txt")
