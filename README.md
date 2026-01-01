 Wearable Gait & Posture Recognition using STM32 and TinyML (No-Hardware Demo)

Overview
This project demonstrates the design of a wearable gait and posture recognition system
using IMU-based sensing, TinyML on-device inference, and battery-powered embedded constraints.

Although no physical hardware is used, the complete **signal processing and TinyML pipeline**
is implemented and validated using a real IMU dataset, ensuring full compatibility
with **STM32 Blue Pill (STM32F103C8)** deployment.

---

System Objectives
- Classify human motion states using IMU data
- Perform window-based feature extraction
- Train a TinyML-compatible ML model
- Ensure STM32 memory, latency, and power feasibility
- Avoid cloud computation (on-device inference)

Motion Classes
The system targets at least three motion states:
- Walking
- Standing / Stationary
- Transitional or abnormal gait states

(Using UCI HAR dataset activities)

 Dataset
UCI Human Activity Recognition (HAR) Dataset

- Sensor: Triaxial accelerometer & gyroscope
- Sampling rate: 50 Hz
- Placement: Waist-mounted smartphone
- Activities: Walking, Sitting, Standing, Lying, etc.

Dataset Link:  
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

 Signal Processing Pipeline
1. IMU signal acquisition (accelerometer X-axis)
2. Windowing:
   - Window size: 128 samples (~2.56 seconds)
   - Overlap: 50%
3. Feature extraction:
   - Mean
   - Standard deviation
4. Feature-label alignment
5. Model training and evaluation

 
 Machine Learning Model
- Model: Decision Tree Classifier
- Reason:
  - Low memory footprint
  - Fast inference
  - Easily convertible to C
  - Ideal for TinyML on STM32

Accuracy achieved: ~46%

 STM32 Deployment Concept
- MCU: STM32F103C8 (Blue Pill)
- Floating-point operations minimized
- Model inference suitable for real-time execution
- LoRa used only for transmitting classification result

 Power & BMS Design (Conceptual)
- Battery: 3.7V Li-Po
- Protection:
  - Over-voltage
  - Under-voltage
  - Over-current
- Charging module: TP4056
- STM32 low-power sleep modes used between inference windows


