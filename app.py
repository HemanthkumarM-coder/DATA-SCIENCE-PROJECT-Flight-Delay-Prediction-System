from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
print("Loading model and encoders...")
model = joblib.load('flight_delay_model.joblib')
le_origin = joblib.load('le_origin.joblib')
le_dest = joblib.load('le_dest.joblib')
le_carrier = joblib.load('le_carrier.joblib')

# Load airport mapping for descriptive names
def load_airport_mapping():
    mapping = {}
    try:
        # OpenFlights format: ID, Name, City, Country, IATA...
        df = pd.read_csv('airports.dat', header=None, 
                         usecols=[1, 3, 4], 
                         names=['name', 'country', 'iata'])
        # Re-reading because IATA is column 4 (0-indexed) or 5 (1-indexed)
        # Actually indexing: 0:ID, 1:Name, 2:City, 3:Country, 4:IATA
        df = pd.read_csv('airports.dat', header=None)
        for _, row in df.iterrows():
            iata = str(row[4]).strip('"')
            if iata and iata != '\\N':
                name = str(row[1]).strip('"')
                country = str(row[3]).strip('"')
                mapping[iata] = f"{name} ({country})"
    except Exception as e:
        print(f"Error loading airport mapping: {e}")
    return mapping

airport_map = load_airport_mapping()

@app.route('/')
def index():
    carriers = sorted(le_carrier.classes_.tolist())
    
    # Process origins with descriptive names
    origins_raw = sorted(le_origin.classes_.tolist())
    origins = [{"code": o, "name": airport_map.get(o, f"{o} Airport")} for o in origins_raw]
    
    # Process destinations with descriptive names
    dests_raw = sorted(le_dest.classes_.tolist())
    dests = [{"code": d, "name": airport_map.get(d, f"{d} Airport")} for d in dests_raw]
    
    return render_template('index.html', carriers=carriers, origins=origins, dests=dests)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        
        # Extract features from form
        month = int(data.get('month'))
        day = int(data.get('day'))
        hour = int(data.get('hour'))
        origin = data.get('origin')
        dest = data.get('dest')
        carrier = data.get('carrier')
        
        # Weather features (defaulting to averages if not provided)
        temp = float(data.get('temp', 55))
        dewp = float(data.get('dewp', 40))
        humid = float(data.get('humid', 60))
        wind_speed = float(data.get('wind_speed', 10))
        precip = float(data.get('precip', 0))
        pressure = float(data.get('pressure', 1013))
        visib = float(data.get('visib', 10))
        
        # Encode categorical features
        origin_enc = le_origin.transform([origin])[0]
        dest_enc = le_dest.transform([dest])[0]
        carrier_enc = le_carrier.transform([carrier])[0]
        
        # Prepare feature vector
        features = np.array([[
            month, day, hour, origin_enc, dest_enc, carrier_enc,
            temp, dewp, humid, wind_speed, precip, pressure, visib
        ]])
        
        # Prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        result = {
            'is_delayed': int(prediction),
            'delay_probability': round(float(probability) * 100, 2),
            'message': 'Flight is likely to be delayed' if prediction == 1 else 'Flight is likely on time'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
