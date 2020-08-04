This repository deals with basic blur detection using Logistic regression.
Train data is divided into 2 folders (sharp and blurry), paste sharp pics into sharp folder and blurry pics
in blurry folder.

Blur is detected by calculating variance of laplace of the image.

On prediction, outputs [1] if blur or [0] if sharp image.

Dependencies:
- Scipy
- scikit image
- scikit learn
- pickle

Use example:
python train.py
python test.py