# 🌊 Project Neptune: AI-Powered IoT Flood Surveillance & Rescue Alert System

![Project Neptune Logo](docs/images/neptune_logo.png)

## 📑 Overview

Project Neptune is an intelligent flood surveillance and rescue alert system that combines the Internet of Things (IoT), AI-based analysis, and long-range communication to monitor and manage flood conditions in real-time. It is designed to prevent disasters and support timely rescue operations during floods or heavy rainfall scenarios.

## 🎯 Key Features

- Real-time water level and environmental monitoring
- AI-powered flood prediction and analysis
- LoRa-based long-range communication
- Automated alert system for authorities and citizens
- AI surveillance for identifying stranded individuals
- Solar-powered remote stations for off-grid operation

## 🏗️ Repository Structure

```
Project-Neptune/
├── src/                  # Source code
│   ├── river_station/    # Code for remote river monitoring stations
│   ├── main_station/     # Main server and data processing code
│   └── ai_surveillance/  # AI models for surveillance and detection
├── docs/                 # Documentation
├── hardware/             # Hardware designs and schematics
├── models/               # Trained AI models
├── utils/                # Utility scripts and tools
└── tests/                # Test scripts
```

## 🚀 Getting Started

### Prerequisites

- Arduino IDE (for microcontroller programming)
- Python 3.8+
- TensorFlow 2.x (for AI models)
- LoRa modules and compatible hardware

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Project-Neptune.git
   ```

2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Upload the Arduino code to your microcontroller:
   - Open `src/river_station/river_station.ino` in Arduino IDE
   - Select your board and port
   - Upload the code

4. Configure the main station:
   - Edit `config.json` with your specific settings
   - Run `python src/main_station/server.py`

## 📊 System Architecture

![System Architecture](docs/images/system_architecture.png)

## 📡 Communication Protocol

Project Neptune uses LoRa (Long Range) technology for its communication layer due to:
- Low power consumption
- Long-range capability (up to 10km in rural areas)
- Resilience in disaster scenarios
- Ability to operate without internet infrastructure

## 🧠 AI Surveillance Module

The AI surveillance system uses computer vision and rule-based logic to:
- Detect water level changes
- Identify potential flood scenarios
- Locate stranded individuals in surveillance zones
- Prioritize rescue operations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or support, please open an issue or contact the project maintainers.
