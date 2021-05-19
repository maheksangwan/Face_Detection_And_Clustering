import os
import cv2
import sys
import json
import face_recognition
import numpy as np
import random

def getDist(pt1, pt2):
    SUM = 0
    for i in range(len(pt1)):
        SUM += (pt1[i] - pt2[i])**2
    return np.sqrt(SUM)

json_list = []
encodingArr = []
imgsDir = sys.argv[1]
imgsDir = os.path.normpath(imgsDir)
numClusters = int(imgsDir.split("_")[1][0])
allImgNames = os.listdir(imgsDir)
face_cascade = cv2.CascadeClassifier('Model_Files/haarcascade_frontalface_default.xml')
for f in allImgNames:
    imgDir = os.path.join(imgsDir, f)
    img = cv2.imread(imgDir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.05,7)
    boxes = []
    for(x,y,w,h) in faces:
        boxes.append((y, x+w, y+h, x))

    encoding = face_recognition.face_encodings(img, boxes)[0]
    encodingArr.append([f,encoding])

temp = random.sample(encodingArr, numClusters)
centroids = []
for val in temp:
    centroids.append(val[1])
iter = 0
while iter<1:
    # Assign Clusters
    clusters = {}
    for encoding in encodingArr:
        minD = 1e10
        centerIdx = -1
        for i, center in enumerate(centroids):
            d = getDist(encoding[1], center)
            if d<minD:
                minD = d
                centerIdx = i

        if centerIdx not in clusters:
            clusters[centerIdx] = []
        clusters[centerIdx].append(encoding)

    # Update Centroids
    for k in clusters:
        SUM = 0
        for val in clusters[k]:
            SUM += val[1]
        centroids[k] += SUM/len(clusters[k])
    iter += 1

for k in range(numClusters):
    outDict = {}
    imgsList = []
    for val in clusters[k]:
        imgsList.append(val[0])
    outDict["cluster no"] = k
    outDict["elements"] = imgsList
    json_list.append(outDict)
with open("clusters.json", 'w') as f:
    json.dump(json_list, f)
