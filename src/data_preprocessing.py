import tensorflow as tf
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from glob import glob
import matplotlib.pyplot as plt


def load_image(path, size):
    image = cv2.imread(path)
    image = cv2.resize(image, (size,size))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   # shape: (size,size,3) -> (size,size,1)
    image = image/255.   # normalize
    return image

def load_data(arr,size):
  t=[]
  i=0
  for path in sorted(arr):
    i+=1
    img=load_image(path,size)

    t.append(img)
    print(i)
  return np.array(t)

all_train_files = glob("Biomedical image segmentation/data/raw/hyper-kvasir-segmented-images/segmented-images/images/*.jpg")
train_mask_files = glob("Biomedical image segmentation/data/raw/hyper-kvasir-segmented-images/segmented-images/masks/*.jpg")
train_mask_files=list(train_gt_files)
train_image_files = list(set(all_train_files) - set(train_mask_files))

size = 128   # image size: 128x128

X = load_data(train_image_files, size)
y=load_data(train_mask_files,size)

##Data Visualization

fig, ax = plt.subplots(5,3, figsize=(10,18))
i = np.random.randint(0,999)

for i in range(5):
    ax[i,0].imshow(X_[i], cmap='gray')
    ax[i,0].set_title('Image')
    ax[i,1].imshow(y[i], cmap='gray')
    ax[i,1].set_title('Mask')
    ax[i,2].imshow(tf.squeeze(y[i]), alpha=0.5, cmap='jet')
    ax[i,2].set_title('Union')
fig.suptitle('Dataset', fontsize=16)
plt.show()

#Changing shape from (1000, 128, 128) to  (1000, 128, 128, 1)
X = np.expand_dims(X, -1)
y = np.expand_dims(y, -1)

#spiliting dataset for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

folder = './processed_data/'
if not os.path.exists(folder):
    os.makedirs(folder)

np.save(folder + 'X_train', Train_img)
np.save(folder + 'X_test', Test_img)
np.save(folder + 'y_train', Train_mask)
np.save(folder + 'y_test', Test_mask)
