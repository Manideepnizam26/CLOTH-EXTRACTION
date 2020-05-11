import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import time

train_images = []
train_output = []
test_images = []
test_output = []

t1 = time.time()



train_images = np.load('./DATA/train_images.npy')
train_output = np.load('./DATA/train_optput.npy')
test_images = np.load('./DATA/test_images.npy')
test_output = np.load('./DATA/test_output.npy')

t2 = time.time()

print(train_images.shape)
print(train_output.shape)
print(test_images.shape)
print(test_output.shape)



print(f"Time took to read npy: {t2-t1} seconds.")