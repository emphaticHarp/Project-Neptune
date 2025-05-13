/*
 * Project Neptune - River Station Module
 * 
 * This code runs on ESP32/Arduino deployed at river monitoring stations.
 * It collects data from various sensors and transmits it via LoRa to the main station.
 * 
 * Hardware Requirements:
 * - ESP32 or Arduino (with LoRa capability)
 * - Ultrasonic Sensor (HC-SR04)
 * - Soil Moisture Sensor
 * - Rain Sensor
 * - LoRa Module (SX1276/SX1278)
 * - Solar Panel & Battery (optional)
 */

#include <SPI.h>
#include <LoRa.h>
#include <ArduinoJson.h>

// Pin Definitions
#define TRIG_PIN 2          // Ultrasonic sensor trigger pin
#define ECHO_PIN 3          // Ultrasonic sensor echo pin
#define SOIL_MOISTURE_PIN A0 // Soil moisture sensor analog pin
#define RAIN_SENSOR_PIN A1   // Rain sensor analog pin
#define WATER_LEVEL_PIN A2   // Water level sensor analog pin

// LoRa Module Pins
#define LORA_SS_PIN 10
#define LORA_RESET_PIN 9
#define LORA_DIO0_PIN 8

// Constants
#define STATION_ID "RIVER_STATION_001"  // Unique ID for this station
#define SAMPLING_INTERVAL 60000         // Data sampling interval (1 minute)
#define ALERT_WATER_LEVEL 70            // Water level threshold for alert (in cm)
#define ALERT_RAIN_INTENSITY 800        // Rain intensity threshold for alert

// Variables
unsigned long lastSampleTime = 0;
float waterLevel = 0.0;
int soilMoisture = 0;
int rainIntensity = 0;
bool alertStatus = false;

void setup() {
  // Initialize Serial
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Project Neptune - River Station Initializing...");
  
  // Initialize pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(SOIL_MOISTURE_PIN, INPUT);
  pinMode(RAIN_SENSOR_PIN, INPUT);
  pinMode(WATER_LEVEL_PIN, INPUT);
  
  // Initialize LoRa
  initLoRa();
  
  Serial.println("River Station Ready!");
}

void loop() {
  // Check if it's time to take a sample
  if (millis() - lastSampleTime >= SAMPLING_INTERVAL) {
    // Read sensor data
    waterLevel = readWaterLevel();
    soilMoisture = readSoilMoisture();
    rainIntensity = readRainIntensity();
    
    // Check alert conditions
    checkAlertConditions();
    
    // Send data via LoRa
    sendDataViaLoRa();
    
    // Update last sample time
    lastSampleTime = millis();
    
    // Print data to Serial (for debugging)
    printSensorData();
  }
  
  // Check for incoming messages from main station
  checkIncomingMessages();
  
  // Small delay to prevent CPU hogging
  delay(100);
}

void initLoRa() {
  Serial.println("Initializing LoRa...");
  
  LoRa.setPins(LORA_SS_PIN, LORA_RESET_PIN, LORA_DIO0_PIN);
  
  if (!LoRa.begin(915E6)) {  // 915MHz frequency (adjust according to your region)
    Serial.println("LoRa initialization failed!");
    while (1);
  }
  
  // Set spreading factor (higher = longer range but slower data rate)
  LoRa.setSpreadingFactor(10);
  
  // Set TX power (higher = longer range but more power consumption)
  LoRa.setTxPower(20);
  
  Serial.println("LoRa initialized successfully!");
}

float readWaterLevel() {
  // Ultrasonic sensor logic to measure water level
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  // Measure the echo time
  long duration = pulseIn(ECHO_PIN, HIGH);
  
  // Calculate distance in cm
  float distance = duration * 0.034 / 2;
  
  // Convert to water level (assuming sensor is mounted above water)
  // This will depend on your specific installation height
  float level = 150 - distance;  // Example: 150cm is the maximum level
  
  return level;
}

int readSoilMoisture() {
  // Read soil moisture sensor
  int moisture = analogRead(SOIL_MOISTURE_PIN);
  
  // Map the raw value to a percentage (0-100%)
  // Adjust these values based on your sensor calibration
  int moisturePercent = map(moisture, 0, 1023, 0, 100);
  
  return moisturePercent;
}

int readRainIntensity() {
  // Read rain sensor
  int rainValue = analogRead(RAIN_SENSOR_PIN);
  
  // Convert to intensity scale (0-1000)
  // Lower values typically indicate more rain
  int intensity = map(rainValue, 0, 1023, 1000, 0);
  
  return intensity;
}

void checkAlertConditions() {
  // Check if water level exceeds threshold
  bool waterLevelAlert = (waterLevel >= ALERT_WATER_LEVEL);
  
  // Check if rain intensity exceeds threshold
  bool rainAlert = (rainIntensity >= ALERT_RAIN_INTENSITY);
  
  // Update alert status
  alertStatus = waterLevelAlert || rainAlert;
  
  // If alert status changed to true, send immediate alert
  if (alertStatus) {
    Serial.println("ALERT CONDITION DETECTED!");
  }
}

void sendDataViaLoRa() {
  // Create JSON document
  StaticJsonDocument<200> doc;
  
  // Add data to JSON
  doc["station_id"] = STATION_ID;
  doc["timestamp"] = millis();
  doc["water_level"] = waterLevel;
  doc["soil_moisture"] = soilMoisture;
  doc["rain_intensity"] = rainIntensity;
  doc["alert"] = alertStatus;
  
  // Serialize JSON to string
  String jsonString;
  serializeJson(doc, jsonString);
  
  // Send packet via LoRa
  Serial.println("Sending data via LoRa...");
  LoRa.beginPacket();
  LoRa.print(jsonString);
  LoRa.endPacket();
  
  Serial.println("Data sent!");
}

void checkIncomingMessages() {
  // Check if there's any incoming packet
  int packetSize = LoRa.parsePacket();
  
  if (packetSize) {
    // Read packet
    String message = "";
    while (LoRa.available()) {
      message += (char)LoRa.read();
    }
    
    Serial.println("Received message: " + message);
    
    // Process commands from main station
    processCommand(message);
  }
}

void processCommand(String command) {
  // Process commands from main station
  // Example: change sampling interval, reset device, etc.
  if (command.startsWith("INTERVAL:")) {
    String intervalStr = command.substring(9);
    unsigned long newInterval = intervalStr.toInt();
    
    if (newInterval > 0) {
      SAMPLING_INTERVAL = newInterval;
      Serial.println("Sampling interval updated to: " + String(SAMPLING_INTERVAL) + "ms");
    }
  }
  else if (command == "RESET") {
    Serial.println("Reset command received. Resetting...");
    // Implement reset logic here
  }
}

void printSensorData() {
  Serial.println("--- Sensor Readings ---");
  Serial.println("Water Level: " + String(waterLevel) + " cm");
  Serial.println("Soil Moisture: " + String(soilMoisture) + "%");
  Serial.println("Rain Intensity: " + String(rainIntensity));
  Serial.println("Alert Status: " + String(alertStatus ? "ALERT!" : "Normal"));
  Serial.println("----------------------");
}
