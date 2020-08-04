import pickle
import os
import numpy as np
from scipy.ndimage import variance
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import laplace
from sklearn.linear_model import LogisticRegression


classifier = LogisticRegression(max_iter=100)
with open('logmodel.pickle', 'rb') as model1:
    classifier = pickle.load(model1)

folder1 = "testdata/"
for filename in os.listdir(folder1):
    print(filename)
    img1 = io.imread(os.path.join(folder1, filename))
    img1 = resize(img1, (480, 640))
    img1 = rgb2gray(img1)
    laplace1 = laplace(img1, ksize=3)
    variance1 = variance(laplace1)
    variance1 = np.array(variance1).reshape(-1,1)
    print(classifier.predict(variance1))
