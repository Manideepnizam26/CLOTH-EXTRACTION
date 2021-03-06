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


files = glob.glob('./DATA/train/*')
print(len(files))

for image_file in files:
    image = cv2.imread(image_file)
    image1, image2 = np.split(image,2,1)
    train_images.append(image1)
    train_output.append(image2)

files = glob.glob('./DATA/test/*')
print(len(files))

for image_file in files:
    image = cv2.imread(image_file)
    image1, image2 = np.split(image,2,1)
    test_images.append(image1)
    test_output.append(image2)

t2 = time.time()

print(f"Time took to read: {t2-t1} seconds.")



np.save('./DATA/train_images.npy', train_images)
np.save('./DATA/train_optput.npy', train_output)
np.save('./DATA/test_images.npy', test_images)
np.save('./DATA/test_output.npy', test_output)

t3 = time.time()

print(f"Time took to save: {t3-t2} seconds.")
    # image1 = image[:256][256:][:3]
    # image2 = image[:][256:][:]
    # print(image1.shape)
    # print(image2.shape)
    # plt.imshow(image1[:,:,[2,1,0]],cmap = 'gray')
    # plt.show()
    # plt.imshow(image2[:,:,[2,1,0]],cmap = 'gray')
    # plt.show()
    # cv2.imshow(image1[:,:,[2,1,0]],cmap = 'gray')
    # plt.imshow(image2)