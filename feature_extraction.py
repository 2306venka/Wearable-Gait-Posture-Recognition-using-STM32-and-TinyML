import numpy as np
import os

BASE_PATH = "../data/UCI_HAR_Dataset/train/Inertial Signals"

def load_signal(filename):
    path = os.path.join(BASE_PATH, filename)
    return np.loadtxt(path)

# Load accelerometer X-axis signal
acc_x = load_signal("body_acc_x_train.txt")

WINDOW_SIZE = 128
STEP = 64

features = []

for i in range(0, acc_x.shape[1] - WINDOW_SIZE, STEP):
    window = acc_x[:, i:i+WINDOW_SIZE]
    mean = np.mean(window)
    std = np.std(window)
    features.append([mean, std])

features = np.array(features)

print("Feature shape:", features.shape)
print("First 5 feature vectors:")
print(features[:5])
