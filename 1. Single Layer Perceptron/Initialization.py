import numpy as np
import matplotlib.pyplot as plt

NUM_FEATURES = 2
NUM_ITER = 100
learning_rate = 0.1

x = np.array([[0,0],[0,1],[1,0],[1,1]], np.float32)
y = np.array([0, 0, 0, 1], np.float32)

# initial weight & bias
W = np.zeros(NUM_FEATURES, np.float32) 
b = np.zeros(1, np.float32)
