# Project Neptune: Hardware Components

## River Station Hardware

### Microcontroller
- **ESP32-WROOM-32D**
  - Dual-core Tensilica LX6 microprocessor
  - 520 KB SRAM
  - Integrated Wi-Fi and Bluetooth
  - Operating voltage: 3.3V
  - 34 programmable GPIO pins

### Sensors
- **HC-SR04 Ultrasonic Distance Sensor** (Water Level)
  - Operating voltage: 5V
  - Range: 2cm - 400cm
  - Resolution: 0.3cm
  - Quantity: 2 per station

- **YF-S201 Water Flow Sensor**
  - Operating voltage: 5-18V DC
  - Maximum current: 15mA (5V)
  - Flow rate range: 1-30L/min
  - Working pressure: ≤1.75MPa
  - Quantity: 1 per station

- **DHT22 Temperature & Humidity Sensor**
  - Operating voltage: 3.3-6V DC
  - Temperature range: -40°C to 80°C
  - Humidity range: 0-100% RH
  - Accuracy: ±0.5°C, ±2% RH
  - Quantity: 1 per station

- **BMP280 Barometric Pressure Sensor**
  - Operating voltage: 1.8-3.6V
  - Pressure range: 300-1100 hPa
  - Relative accuracy: ±0.12 hPa
  - Absolute accuracy: ±1 hPa
  - Quantity: 1 per station

### Communication
- **RFM95W LoRa Transceiver Module**
  - Frequency: 868/915 MHz
  - Range: Up to 2km in urban areas, 10km+ line of sight
  - Interface: SPI
  - Operating voltage: 3.3V
  - Transmit power: +20 dBm

- **SIM800L GSM/GPRS Module** (Backup communication)
  - Quad-band 850/900/1800/1900MHz
  - Operating voltage: 3.4-4.4V
  - GPRS data downlink transfer: 85.6 kbps
  - GPRS data uplink transfer: 85.6 kbps

### Power System
- **5W Solar Panel**
  - Peak power: 5W
  - Operating voltage: 6V
  - Dimensions: 250mm × 180mm × 3mm

- **TP4056 Lithium Battery Charging Module**
  - Input voltage: 4.5-5.5V
  - Full charge voltage: 4.2V
  - Charging current: 1A (adjustable)

- **18650 Lithium-ion Battery**
  - Capacity: 3400mAh
  - Nominal voltage: 3.7V
  - Quantity: 2 per station (in series)

- **LM2596 DC-DC Buck Converter**
  - Input voltage: 4.5-40V
  - Output voltage: 1.25-37V (adjustable)
  - Output current: Max 3A

### Enclosure
- **Waterproof Junction Box**
  - IP67 rated
  - Material: ABS plastic
  - Dimensions: 200mm × 120mm × 75mm
  - Mounting brackets included

## Main Station Hardware

### Server
- **Raspberry Pi 4 Model B**
  - Processor: Quad-core Cortex-A72 (ARM v8) 64-bit @ 1.5GHz
  - RAM: 4GB LPDDR4-3200
  - Storage: 64GB microSD card (Class 10)
  - Connectivity: Gigabit Ethernet, dual-band Wi-Fi, Bluetooth 5.0

### Communication
- **RFM95W LoRa Gateway**
  - Based on RAK2245 Pi HAT
  - 8-channel LoRa concentrator
  - Frequency: 868/915 MHz
  - Range: Up to 15km line of sight

- **4G LTE Modem**
  - Huawei E3372 USB Dongle
  - Fallback internet connectivity

### Power Supply
- **UPS HAT for Raspberry Pi**
  - Input: 5V/3A
  - Battery: 18650 Lithium-ion (3400mAh)
  - Provides up to 9 hours of backup power

## AI Surveillance Hardware

### Camera System
- **Raspberry Pi Camera Module V2**
  - 8MP Sony IMX219 sensor
  - 1080p30 video
  - Fixed focus lens
  - Quantity: 2 per surveillance point

- **Weatherproof Camera Housing**
  - IP66 rated
  - Operating temperature: -20°C to 60°C
  - Mounting brackets included

### Processing Unit
- **NVIDIA Jetson Nano Developer Kit**
  - CPU: Quad-core ARM A57 @ 1.43 GHz
  - GPU: 128-core NVIDIA Maxwell architecture
  - Memory: 4GB 64-bit LPDDR4
  - Storage: 64GB microSD card

### Power Supply
- **12V 5A Power Adapter**
  - Input: 100-240V AC
  - Output: 12V DC, 5A
  - Barrel jack connector
