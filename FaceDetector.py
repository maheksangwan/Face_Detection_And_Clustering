import os
import cv2
import sys
import json

json_list = []
imgsDir = str(sys.argv[1])+"/images"
imgsDir = os.path.normpath(imgsDir)
allImgNames = os.listdir(imgsDir)
face_cascade = cv2.CascadeClassifier('Model_Files/haarcascade_frontalface_default.xml')

for f in allImgNames:
    imgDir = os.path.join(imgsDir, f)
    img = cv2.imread(imgDir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.05, 7)
    for(x,y,w,h) in faces:
        element = {"iname": f, "bbox": [int(x), int(y), int(w), int(h)]}
        json_list.append(element)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
with open("results.json", 'w') as f:
    json.dump(json_list, f)
