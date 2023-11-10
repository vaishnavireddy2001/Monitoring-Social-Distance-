import cv2
import numpy as np

vid=cv2.VideoCapture('/Users/lazycoder/Desktop/IEEE/video.mp4')
wht = 320


classFile = '/Users/lazycoder/Desktop/IEEE/coco.names.txt'
classNames = []
confThreshold = 0.5
nmsThreshold = 0.3 # the more less it is, the more powerfull nms becomes


with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = '/Users/lazycoder/Desktop/IEEE/YOLO/yolov3.cfg'
modelWeights = '/Users/lazycoder/Desktop/IEEE/YOLO/yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findobjects(outputs,img):
     hT, wT, cT = img.shape
     bbox = [] #will contain x,y,w &h
     classIds = []
     confs = []

     for outputs in outputs:
         for det in outputs: #we will call each box as a detection.
             scores = det[5:] #removing top 5 outputs
             classId = np.argmax(scores)
             confidence = scores[classId]
             if confidence > confThreshold:
                if classNames[classId]=="person":
                     w,h = int(det[2]*wT) , int(det[3]*hT) #mutiplying as det[2] and so are in %.
                     x,y = int((det[0]*wT)- w/2), int((det[1]*hT)- h/2)
                     bbox.append([x,y,w,h])
                     classIds.append(classId)
                     confs.append(float(confidence))

     indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
     #print(indies)  ans= [[0]]
     for i in indices:
         i = i[0]
         box = bbox[i]
         x,y,w,h = box[0], box[1], box[2], box[3]
         cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,255),2)


while True:
    success, img = vid.read()
    blob = cv2.dnn.blobFromImage(img,1/255,(wht,wht),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    #print(layerNames) this will print out layer name
    outputNames=[layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(net.getUnconnectedOutLayers()) #this will print out index, but remember index start from 1 in this, so we have to -1(as we index from 0) to get real index.
    #print(outputNames)

    outputs = net.forward(outputNames)
    #print(len(outputs)) # answer 3; 3 output layers
    # print(outputs[0].shape) #300,85
    # print(outputs[1].shape) #1200,85
    # print(outputs[2].shape) #4800, 85
    # print(outputs[0][0].shape) # to print first row of first output layer. total 300 rows in first layer.
    findobjects(outputs, img)
   
   
    cv2.imshow('Output',img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
         break
