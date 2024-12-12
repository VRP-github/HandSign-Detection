import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")


offset = 20
imgSize = 300
folder = "Images/C"

labels = ["A","B","C","OK"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgcrop = img[y-offset:y+h+offset, x-offset:x + w+offset]

        imgCropShape = imgcrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgcrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            # Move this line inside the aspect ratio condition
            prediction, index = classifier.getPrediction(imgWhite, draw = False)
            print(prediction, index)
            

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgcrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal+hGap, :] = imgResize
            # Move this line inside the aspect ratio condition
            prediction, index = classifier.getPrediction(imgWhite, draw = False)
            #print(prediction, index)
        
        cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        cv2.rectangle(imgOutput,(x,y),(x+w,y+h),(255,0,255),4)
        cv2.putText(imgOutput,"Wave a sign at some distance from Camera!",(10,10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
        cv2.putText(imgOutput,"Hand_Sign_Detection",(200,450),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),2)

        cv2.imshow("Image_Crop", imgcrop)
        cv2.imshow("Image_White", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)


