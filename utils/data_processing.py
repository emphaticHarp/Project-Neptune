"""
Project Neptune - Data Processing Utilities
This module contains utility functions for processing sensor data,
calculating statistics, and preparing data for visualization.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("neptune_data_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("neptune_utils")

def load_sensor_data(file_path):
    """
    Load sensor data from CSV file into pandas DataFrame
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing sensor data
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        logger.info(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        return None

def calculate_rolling_average(df, column, window=6):
    """
    Calculate rolling average for a specific column
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to calculate rolling average for
        window (int): Window size in hours
        
    Returns:
        pd.Series: Series containing rolling average values
    """
    try:
        # Ensure DataFrame is sorted by timestamp
        df = df.sort_values('timestamp')
        
        # Calculate rolling average
        rolling_avg = df[column].rolling(window).mean()
        
        logger.info(f"Calculated {window}-hour rolling average for {column}")
        return rolling_avg
    except Exception as e:
        logger.error(f"Error calculating rolling average: {str(e)}")
        return None

def detect_anomalies(df, column, threshold=3.0):
    """
    Detect anomalies in sensor data using z-score method
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for anomalies
        threshold (float): Z-score threshold for anomaly detection
        
    Returns:
        pd.DataFrame: DataFrame with anomaly flags
    """
    try:
        # Calculate z-score
        mean = df[column].mean()
        std = df[column].std()
        z_scores = (df[column] - mean) / std
        
        # Flag anomalies
        df['anomaly'] = (abs(z_scores) > threshold)
        
        anomaly_count = df['anomaly'].sum()
        logger.info(f"Detected {anomaly_count} anomalies in {column}")
        
        return df
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        return df

def interpolate_missing_values(df, column):
    """
    Interpolate missing values in a time series
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to interpolate
        
    Returns:
        pd.Series: Series with interpolated values
    """
    try:
        # Count missing values before interpolation
        missing_before = df[column].isna().sum()
        
        # Interpolate missing values
        interpolated = df[column].interpolate(method='time')
        
        # Count remaining missing values
        missing_after = interpolated.isna().sum()
        
        logger.info(f"Interpolated {missing_before - missing_after} missing values in {column}")
        return interpolated
    except Exception as e:
        logger.error(f"Error interpolating missing values: {str(e)}")
        return df[column]

def calculate_flood_risk(water_level, rainfall, flow_rate):
    """
    Calculate flood risk based on multiple parameters
    
    Args:
        water_level (float): Current water level in cm
        rainfall (float): Rainfall in mm over past 24 hours
        flow_rate (float): Water flow rate in L/min
        
    Returns:
        tuple: (risk_level, risk_score)
            risk_level (str): 'Low', 'Moderate', 'High', or 'Severe'
            risk_score (float): Numerical risk score between 0-100
    """
    try:
        # Normalize inputs to 0-1 scale based on typical ranges
        norm_level = min(1.0, water_level / 500.0)  # Assuming 500cm is max level
        norm_rainfall = min(1.0, rainfall / 100.0)  # Assuming 100mm is heavy rainfall
        norm_flow = min(1.0, flow_rate / 30.0)      # Assuming 30L/min is high flow
        
        # Calculate weighted risk score (0-100)
        weights = [0.5, 0.3, 0.2]  # Weights for level, rainfall, flow
        risk_score = 100 * (weights[0] * norm_level + 
                           weights[1] * norm_rainfall + 
                           weights[2] * norm_flow)
        
        # Determine risk level
        if risk_score < 25:
            risk_level = 'Low'
        elif risk_score < 50:
            risk_level = 'Moderate'
        elif risk_score < 75:
            risk_level = 'High'
        else:
            risk_level = 'Severe'
            
        logger.info(f"Calculated flood risk: {risk_level} ({risk_score:.1f})")
        return (risk_level, risk_score)
    except Exception as e:
        logger.error(f"Error calculating flood risk: {str(e)}")
        return ('Unknown', 0.0)

def prepare_dashboard_data(df, station_id):
    """
    Prepare data for dashboard visualization
    
    Args:
        df (pd.DataFrame): Input DataFrame with sensor readings
        station_id (str): Station identifier
        
    Returns:
        dict: Formatted data for dashboard
    """
    try:
        # Get latest readings
        latest = df.iloc[-1]
        
        # Calculate 24h change
        if len(df) > 24:
            prev_24h = df.iloc[-25]
            water_level_change = latest['water_level'] - prev_24h['water_level']
        else:
            water_level_change = 0
            
        # Calculate flood risk
        risk_level, risk_score = calculate_flood_risk(
            latest['water_level'],
            latest.get('rainfall_24h', 0),
            latest.get('flow_rate', 0)
        )
        
        # Format data for dashboard
        dashboard_data = {
            'station_id': station_id,
            'timestamp': latest['timestamp'].isoformat(),
            'current_readings': {
                'water_level': round(latest['water_level'], 1),
                'flow_rate': round(latest.get('flow_rate', 0), 2),
                'temperature': round(latest.get('temperature', 0), 1),
                'humidity': round(latest.get('humidity', 0), 1),
                'rainfall_24h': round(latest.get('rainfall_24h', 0), 1)
            },
            'changes': {
                'water_level_24h': round(water_level_change, 1)
            },
            'risk_assessment': {
                'level': risk_level,
                'score': round(risk_score, 1)
            },
            'historical': {
                'water_level_48h': df['water_level'].tail(48).tolist(),
                'timestamps_48h': [ts.isoformat() for ts in df['timestamp'].tail(48)]
            }
        }
        
        logger.info(f"Prepared dashboard data for station {station_id}")
        return dashboard_data
    except Exception as e:
        logger.error(f"Error preparing dashboard data: {str(e)}")
        return {}

def export_to_json(data, output_file):
    """
    Export data to JSON file
    
    Args:
        data (dict): Data to export
        output_file (str): Path to output JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Successfully exported data to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error exporting data to {output_file}: {str(e)}")
        return False

def generate_alert_message(station_id, risk_level, water_level, trend):
    """
    Generate alert message based on flood risk
    
    Args:
        station_id (str): Station identifier
        risk_level (str): Risk level ('Low', 'Moderate', 'High', 'Severe')
        water_level (float): Current water level in cm
        trend (str): Trend direction ('rising', 'falling', 'stable')
        
    Returns:
        str: Formatted alert message
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        if risk_level == 'Low':
            return f"INFO: Station {station_id} reports normal conditions at {timestamp}. Water level: {water_level}cm, trend: {trend}."
            
        elif risk_level == 'Moderate':
            return f"ADVISORY: Station {station_id} reports elevated water levels at {timestamp}. Current level: {water_level}cm, trend: {trend}. Monitor conditions."
            
        elif risk_level == 'High':
            return f"WARNING: Station {station_id} reports high water levels at {timestamp}. Current level: {water_level}cm, trend: {trend}. Potential flooding possible."
            
        elif risk_level == 'Severe':
            return f"EMERGENCY: Station {station_id} reports severe flooding risk at {timestamp}. Current level: {water_level}cm, trend: {trend}. Immediate action required."
            
        else:
            return f"ALERT: Station {station_id} reports unusual conditions at {timestamp}. Water level: {water_level}cm, trend: {trend}."
            
    except Exception as e:
        logger.error(f"Error generating alert message: {str(e)}")
        return f"ALERT: Flood monitoring system reports potential risk at Station {station_id}."

def calculate_water_level_trend(df, hours=3):
    """
    Calculate water level trend over specified time period
    
    Args:
        df (pd.DataFrame): Input DataFrame with water level readings
        hours (int): Number of hours to analyze for trend
        
    Returns:
        tuple: (trend, change_rate)
            trend (str): 'rising', 'falling', or 'stable'
            change_rate (float): Rate of change in cm/hour
    """
    try:
        # Get recent data
        recent = df.tail(hours + 1)
        
        if len(recent) < 2:
            return ('unknown', 0.0)
            
        # Calculate linear regression
        y = recent['water_level'].values
        x = np.arange(len(y))
        
        # Simple linear regression
        slope, _ = np.polyfit(x, y, 1)
        
        # Convert slope to cm/hour
        readings_per_hour = 12  # Assuming 5-minute intervals
        change_rate = slope * readings_per_hour
        
        # Determine trend
        if abs(change_rate) < 0.5:  # Less than 0.5 cm/hour change
            trend = 'stable'
        elif change_rate > 0:
            trend = 'rising'
        else:
            trend = 'falling'
            
        logger.info(f"Water level trend: {trend} at {abs(change_rate):.2f} cm/hour")
        return (trend, change_rate)
    except Exception as e:
        logger.error(f"Error calculating water level trend: {str(e)}")
        return ('unknown', 0.0)

# Example usage
if __name__ == "__main__":
    # This is just for demonstration
    sample_data = {
        'timestamp': [datetime.now() - timedelta(hours=i) for i in range(48, 0, -1)],
        'water_level': [120 + i + np.random.normal(0, 2) for i in range(48)],
        'flow_rate': [15 + np.random.normal(0, 1) for _ in range(48)],
        'temperature': [25 + np.random.normal(0, 0.5) for _ in range(48)],
        'humidity': [80 + np.random.normal(0, 2) for _ in range(48)],
        'rainfall_24h': [25 + np.random.normal(0, 0.2) for _ in range(48)]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Demo of utility functions
    print("Sample data processing:")
    print("-" * 50)
    
    # Calculate rolling average
    df['water_level_avg'] = calculate_rolling_average(df, 'water_level', window=6)
    print(f"Rolling average calculated: {not df['water_level_avg'].isna().all()}")
    
    # Detect anomalies
    df = detect_anomalies(df, 'water_level', threshold=2.0)
    print(f"Anomalies detected: {df['anomaly'].sum()}")
    
    # Calculate trend
    trend, rate = calculate_water_level_trend(df, hours=6)
    print(f"Water level trend: {trend} at {abs(rate):.2f} cm/hour")
    
    # Calculate risk
    latest = df.iloc[-1]
    risk_level, risk_score = calculate_flood_risk(
        latest['water_level'],
        latest['rainfall_24h'],
        latest['flow_rate']
    )
    print(f"Current risk assessment: {risk_level} ({risk_score:.1f})")
    
    # Generate alert message
    alert = generate_alert_message("RS001", risk_level, latest['water_level'], trend)
    print(f"Alert message: {alert}")
    
    # Prepare dashboard data
    dashboard_data = prepare_dashboard_data(df, "RS001")
    print(f"Dashboard data prepared: {len(dashboard_data) > 0}")
    
    # Export to JSON
    success = export_to_json(dashboard_data, "dashboard_data_example.json")
    print(f"Data export successful: {success}")
