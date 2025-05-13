# Project Neptune: AI Model Architecture

## Flood Detection Model

### Architecture: YOLOv4-tiny
- **Framework**: Darknet
- **Input Size**: 416x416 pixels
- **Backbone**: CSPDarknet53-tiny
- **Feature Extractor**: SPP, PAN
- **Output Layers**: 2 detection layers at different scales
- **Anchor Boxes**: 6 predefined anchor boxes

### Training Details
- **Dataset**: Custom flood imagery dataset (5,000+ labeled images)
- **Augmentation**: Random crop, flip, rotation, color jitter, mosaic
- **Epochs**: 300
- **Batch Size**: 64
- **Optimizer**: Adam with cosine annealing learning rate
- **Initial Learning Rate**: 0.001
- **IoU Threshold**: 0.5
- **Confidence Threshold**: 0.25

### Performance Metrics
- **mAP@0.5**: 0.87
- **Precision**: 0.91
- **Recall**: 0.84
- **Inference Time**: 15ms on Jetson Nano

### Classes
1. Minor Flooding
2. Moderate Flooding
3. Severe Flooding
4. Flash Flood
5. Urban Flooding

## Person Detection Model

### Architecture: MobileNetV2-SSD
- **Framework**: TensorFlow Lite
- **Input Size**: 300x300 pixels
- **Backbone**: MobileNetV2
- **Feature Extractor**: FPN
- **Output**: Single-shot multibox detection
- **Quantization**: 8-bit integer quantization

### Training Details
- **Dataset**: Combination of COCO person class and custom flood rescue imagery
- **Fine-tuning**: Transfer learning from COCO-pretrained model
- **Augmentation**: Random crop, flip, rotation, brightness, contrast
- **Epochs**: 150
- **Batch Size**: 32
- **Optimizer**: SGD with momentum
- **Initial Learning Rate**: 0.01
- **Learning Rate Schedule**: Step decay
- **IoU Threshold**: 0.5
- **Confidence Threshold**: 0.35

### Performance Metrics
- **mAP@0.5**: 0.82
- **Precision**: 0.88
- **Recall**: 0.79
- **Inference Time**: 25ms on Jetson Nano

### Detection Categories
1. Person (standing)
2. Person (in water)
3. Person (on roof/elevated surface)
4. Group of people

## Water Level Prediction Model

### Architecture: LSTM-FCN (Long Short-Term Memory Fully Convolutional Network)
- **Framework**: TensorFlow
- **Input**: Time series data (24 hours of sensor readings)
- **LSTM Layer**: 128 units
- **Conv1D Layers**: 3 layers (64, 128, 128 filters)
- **Dropout Rate**: 0.3
- **Output**: Water level prediction for next 6 hours

### Training Details
- **Dataset**: 2 years of historical water level data
- **Features**: Water level, flow rate, rainfall, temperature, upstream readings
- **Sequence Length**: 24 hours
- **Prediction Horizon**: 6 hours
- **Train/Val/Test Split**: 70/15/15
- **Epochs**: 100
- **Batch Size**: 32
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Early Stopping**: Patience of 15 epochs

### Performance Metrics
- **RMSE**: 5.2cm
- **MAE**: 3.8cm
- **R²**: 0.92
- **Inference Time**: 8ms on Raspberry Pi 4

## Model Deployment

### Edge Deployment (Jetson Nano)
- **Runtime**: TensorRT for optimized inference
- **Batch Processing**: Single image processing
- **Memory Footprint**: <1GB RAM
- **Power Consumption**: <10W

### Server Deployment (Raspberry Pi 4)
- **Runtime**: TensorFlow Lite
- **Scheduling**: Periodic inference (every 5 minutes)
- **Memory Footprint**: <512MB RAM
- **Integration**: REST API for dashboard access

## Model Update Pipeline
- **Frequency**: Quarterly updates
- **Continuous Learning**: New data incorporated monthly
- **Validation**: Automated testing against holdout dataset
- **Deployment**: OTA updates to edge devices
