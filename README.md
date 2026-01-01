Wearable Gait Recognition using TinyML

Overview
This project implements a wearable gait recognition system using sensor data
and a lightweight machine learning model suitable for TinyML applications.

The system extracts statistical features from accelerometer signals and trains
a Decision Tree classifier for gait classification.


Dataset
Dataset used: UCI Human Activity Recognition (HAR) Dataset

Due to size constraints, the dataset is NOT included in this repository.

Dataset link:
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

Required path structure:
UCI HAR Dataset/train/Inertial Signals/
gait_project/

Project Structure
│
├── scripts/
│ ├── feature_extraction.py
│ ├── train_model.py
│
├── model/
│ └── gait_model.pkl
│
├── report/
│ └── Wearable_Gait_TinyML_Report.pdf
│
└── README.md

 Feature Extraction
- Sliding window size: 128
- Step size: 64
- Features extracted:
  - Mean
  - Standard Deviation


 Model Training
- Algorithm: Decision Tree Classifier
- Library: scikit-learn
- Training accuracy: ~46%

The trained model is saved as `gait_model.pkl`.


How to Run
1. Download and extract the UCI HAR dataset
2. Place it in the correct directory
3. Run feature extraction:

4. Train the model:
 Tools Used
- Python 3.14
- NumPy
- scikit-learn

Author
Venkatesh C


