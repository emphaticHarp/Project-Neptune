#!/usr/bin/env python3
"""
Project Neptune - Main Station Server

This script runs on the main station (server/laptop) and:
1. Receives data from river stations via LoRa
2. Processes and analyzes the data
3. Triggers alerts when necessary
4. Provides a web interface for monitoring
5. Interfaces with the AI surveillance module

Requirements:
- Python 3.8+
- PyLoRa
- Flask
- NumPy
- Pandas
- Matplotlib (for visualization)
"""

import os
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any

# For LoRa communication
try:
    import SX127x
    from SX127x.LoRa import LoRa
    from SX127x.board_config import BOARD
except ImportError:
    print("Warning: LoRa libraries not found. Running in simulation mode.")
    SIMULATION_MODE = True
else:
    SIMULATION_MODE = False

# For web interface
from flask import Flask, render_template, jsonify, request

# For data processing
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Import AI surveillance module
import sys
sys.path.append('../ai_surveillance')
from flood_detection import FloodDetector
from person_detection import PersonDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("neptune_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NeptuneServer")

# Constants
CONFIG_FILE = "config.json"
DATA_DIR = "data"
ALERT_THRESHOLD = 0.7  # Probability threshold for flood alert

# Global variables
stations_data = {}  # Store latest data from all stations
flood_alerts = {}   # Store active flood alerts
person_alerts = {}  # Store detected persons in need of rescue

# Initialize Flask app
app = Flask(__name__)

# Load configuration
def load_config():
    """Load configuration from config.json file"""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded successfully from {CONFIG_FILE}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        # Return default configuration
        return {
            "lora": {
                "frequency": 915,
                "spreading_factor": 10,
                "power": 20
            },
            "stations": [],
            "alert": {
                "sms_enabled": False,
                "email_enabled": True,
                "recipients": ["emergency@example.com"]
            },
            "surveillance": {
                "enabled": True,
                "camera_sources": []
            }
        }

# Initialize LoRa
def initialize_lora(config):
    """Initialize LoRa communication"""
    if SIMULATION_MODE:
        logger.warning("Running in simulation mode. LoRa hardware not available.")
        return None
    
    try:
        # Configure LoRa board
        BOARD.setup()
        
        # Create LoRa object
        lora = LoRa(verbose=False)
        
        # Configure LoRa parameters
        lora.set_mode(LoRa.STDBY)
        lora.set_freq(config["lora"]["frequency"])
        lora.set_spreading_factor(config["lora"]["spreading_factor"])
        lora.set_pa_config(pa_select=1, max_power=15, output_power=config["lora"]["power"])
        
        logger.info("LoRa initialized successfully")
        return lora
    except Exception as e:
        logger.error(f"Error initializing LoRa: {e}")
        return None

# Process received data
def process_data(data):
    """Process data received from river stations"""
    try:
        # Parse JSON data
        station_data = json.loads(data)
        
        # Extract station ID
        station_id = station_data.get("station_id")
        
        if not station_id:
            logger.warning(f"Received data without station ID: {data}")
            return
        
        # Add timestamp
        station_data["received_time"] = datetime.now().isoformat()
        
        # Store data
        stations_data[station_id] = station_data
        
        # Save data to file
        save_data_to_file(station_id, station_data)
        
        # Check for alert conditions
        check_alert_conditions(station_id, station_data)
        
        logger.info(f"Processed data from station {station_id}")
    except Exception as e:
        logger.error(f"Error processing data: {e}")

def save_data_to_file(station_id, data):
    """Save station data to file for historical records"""
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Create station directory if it doesn't exist
    station_dir = os.path.join(DATA_DIR, station_id)
    os.makedirs(station_dir, exist_ok=True)
    
    # Create filename based on date
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(station_dir, f"{date_str}.jsonl")
    
    # Append data to file
    with open(filename, 'a') as f:
        f.write(json.dumps(data) + '\n')

# AI-based flood prediction
def predict_flood_probability(station_id, data):
    """
    Use machine learning to predict flood probability
    In a real implementation, this would use a trained model
    """
    # For demonstration, using a simple rule-based approach
    # In a real implementation, this would use the trained model from models/
    
    water_level = data.get("water_level", 0)
    rain_intensity = data.get("rain_intensity", 0)
    soil_moisture = data.get("soil_moisture", 0)
    
    # Simple weighted formula for demonstration
    # In reality, this would use a trained ML model
    probability = (0.5 * (water_level / 100) + 
                   0.3 * (rain_intensity / 1000) + 
                   0.2 * (soil_moisture / 100))
    
    return min(1.0, max(0.0, probability))

def check_alert_conditions(station_id, data):
    """Check if alert conditions are met and trigger alerts if necessary"""
    # Check if station already has an alert
    has_existing_alert = station_id in flood_alerts
    
    # Get alert status from data
    direct_alert = data.get("alert", False)
    
    # Predict flood probability using AI
    flood_probability = predict_flood_probability(station_id, data)
    
    # Determine if alert should be triggered
    should_alert = direct_alert or (flood_probability >= ALERT_THRESHOLD)
    
    if should_alert:
        if not has_existing_alert:
            # New alert
            alert_info = {
                "station_id": station_id,
                "start_time": datetime.now().isoformat(),
                "probability": flood_probability,
                "data": data
            }
            flood_alerts[station_id] = alert_info
            
            # Trigger alert notification
            send_alert_notification(alert_info)
            
            logger.warning(f"FLOOD ALERT triggered for station {station_id} with probability {flood_probability:.2f}")
        else:
            # Update existing alert
            flood_alerts[station_id]["probability"] = flood_probability
            flood_alerts[station_id]["data"] = data
            flood_alerts[station_id]["last_update"] = datetime.now().isoformat()
    elif has_existing_alert:
        # Clear alert
        del flood_alerts[station_id]
        logger.info(f"Flood alert cleared for station {station_id}")

def send_alert_notification(alert_info):
    """Send alert notifications via configured channels (SMS, email, etc.)"""
    config = load_config()
    
    # Format alert message
    station_id = alert_info["station_id"]
    probability = alert_info["probability"]
    water_level = alert_info["data"].get("water_level", "N/A")
    
    message = f"FLOOD ALERT: Station {station_id} reports flood risk with {probability:.2f} probability. Water level: {water_level}cm."
    
    # Send SMS if enabled
    if config["alert"]["sms_enabled"]:
        # In a real implementation, this would use an SMS API
        logger.info(f"Would send SMS: {message}")
    
    # Send email if enabled
    if config["alert"]["email_enabled"]:
        recipients = config["alert"]["recipients"]
        # In a real implementation, this would use an email API
        logger.info(f"Would send email to {recipients}: {message}")

# Flask routes for web interface
@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('index.html')

@app.route('/api/stations')
def api_stations():
    """Return data for all stations"""
    return jsonify(stations_data)

@app.route('/api/alerts')
def api_alerts():
    """Return all active alerts"""
    return jsonify({
        "flood_alerts": flood_alerts,
        "person_alerts": person_alerts
    })

@app.route('/api/station/<station_id>')
def api_station(station_id):
    """Return data for a specific station"""
    if station_id in stations_data:
        return jsonify(stations_data[station_id])
    else:
        return jsonify({"error": "Station not found"}), 404

@app.route('/api/history/<station_id>')
def api_history(station_id):
    """Return historical data for a specific station"""
    try:
        # Get date parameter, default to today
        date_str = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
        
        # Construct filename
        filename = os.path.join(DATA_DIR, station_id, f"{date_str}.jsonl")
        
        if not os.path.exists(filename):
            return jsonify({"error": "No data available for this date"}), 404
        
        # Read data from file
        data = []
        with open(filename, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        return jsonify({"error": str(e)}), 500

# Simulation thread for testing without actual hardware
def simulation_thread():
    """Simulate data from river stations for testing"""
    logger.info("Starting simulation thread")
    
    # Define simulated stations
    stations = [
        {"id": "RIVER_STATION_001", "base_water_level": 30, "base_soil_moisture": 40},
        {"id": "RIVER_STATION_002", "base_water_level": 25, "base_soil_moisture": 35},
        {"id": "RIVER_STATION_003", "base_water_level": 20, "base_soil_moisture": 30}
    ]
    
    # Simulation loop
    while True:
        for station in stations:
            # Generate simulated data
            water_level = station["base_water_level"] + np.random.normal(0, 5)
            soil_moisture = station["base_soil_moisture"] + np.random.normal(0, 10)
            rain_intensity = max(0, np.random.normal(200, 150))
            
            # Create data packet
            data = {
                "station_id": station["id"],
                "timestamp": int(time.time() * 1000),
                "water_level": water_level,
                "soil_moisture": soil_moisture,
                "rain_intensity": rain_intensity,
                "alert": water_level > 50 or rain_intensity > 500
            }
            
            # Process the simulated data
            process_data(json.dumps(data))
        
        # Sleep for a while
        time.sleep(10)

# Main function
def main():
    """Main function to start the server"""
    logger.info("Starting Project Neptune Main Station Server")
    
    # Create necessary directories
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Load configuration
    config = load_config()
    
    # Initialize LoRa
    lora = initialize_lora(config)
    
    # Start simulation thread if in simulation mode
    if SIMULATION_MODE:
        sim_thread = threading.Thread(target=simulation_thread)
        sim_thread.daemon = True
        sim_thread.start()
    
    # Start AI surveillance if enabled
    if config["surveillance"]["enabled"]:
        try:
            # Initialize AI detectors
            flood_detector = FloodDetector()
            person_detector = PersonDetector()
            
            # Start surveillance thread
            # This would be implemented in a real system
            logger.info("AI surveillance module initialized")
        except Exception as e:
            logger.error(f"Error initializing AI surveillance: {e}")
    
    # Start Flask web server
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()
