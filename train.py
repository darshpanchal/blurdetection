import numpy as np
from scipy.ndimage import variance
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import laplace
from skimage.filters import sobel
from sklearn.linear_model import LogisticRegression
import os, pickle

imgvariance = []
y = []

trainfolder = "traindata"

for folder in os.listdir(trainfolder):
    for filename in os.listdir(trainfolder + "/" + folder):
        if "sharp" in folder:
            label = 0
        elif "blurry" in folder:
            label = 1
        img = io.imread(os.path.join(trainfolder, folder, filename))
        img = resize(img, (480, 640))
        img = rgb2gray(img)
        laplace1 = laplace(img, ksize=3)
        variance1 = variance(laplace1)
        imgvariance.append(variance1)
        y.append(label)


imgvariance = np.array(imgvariance).reshape(-1,1)
y = np.array(y).reshape(-1,1)

classifier = LogisticRegression(max_iter=100)
classifier.fit(imgvariance, y)

with open('logmodel.pickle', 'wb') as model:
    pickle.dump(classifier, model, protocol=pickle.HIGHEST_PROTOCOL)
