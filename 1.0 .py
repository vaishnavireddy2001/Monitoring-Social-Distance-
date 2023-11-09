import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

vid=cv2.VideoCapture('/Users/lazycoder/Desktop/IEEE/video.mp4')
#img=cv2.imread('/Users/lazycoder/Desktop/IEEE/Screenshot 2020-11-06 at 7.50.01 PM.png')
wht = 320



classFile = '/Users/lazycoder/Desktop/IEEE/coco.names.txt'
classNames = []
confThreshold = 0.5
nmsThreshold = 0.3 # the more less it is, the more powerfull nms becomes




with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = '/Users/lazycoder/Desktop/IEEE/YOLO/yolov3-tiny.cfg'
modelWeights = '/Users/lazycoder/Desktop/IEEE/YOLO/yolov3-tiny.weights'

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
     

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]

        cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0),2)
        cv2.circle(img, (int(x+w/2), int(y+h/2)), 2, (0, 0, 255), 2) #locating center of each pedestrian
        total.append([x,y,w,h])
        
        

    i=len(indices)
    while i>0:
        j=len(indices)
        #safe_count=0
        #risk_count=0
        while j>i:
            #print(data[i-1],data[j-1])
             
            box1=bbox[indices[i-1][0]]
            x1,y1,w1,h1 = box1[0], box1[1], box1[2], box1[3]

            box2=bbox[indices[j-1][0]]
            x2,y2,w2,h2 = box2[0], box2[1], box2[2], box2[3]
             
                
            #distance formula:
            if(((int(x1+w1/2)-int(x2+w2/2))**2+(int(y1+h1/2)-int(y2+h2/2))**2)**1/2 < (x1+w1)*4):
                cv2.line(img,(int(x1+w1/2), int(y1+h1/2)),(int(x2+w2/2), int(y2+h2/2)),(255,0,0),1)
                cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1), (0,0,255),2)
                cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2), (0,0,255),2)
                red.append([x1,y1,w1,h1])
                red.append([x2,y2,w2,h2])
                
            j=j-1
        i=i-1

        
def func(pct, allvalues): 
    absolute = int(pct / 100.*np.sum(allvalues)) 
    return "{:.1f}%\n({:d} g)".format(pct, absolute) 



     
         
while True:
    success, img = vid.read()
    blob = cv2.dnn.blobFromImage(img,1/255,(wht,wht),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()

    outputNames=[layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)

    hT, wT, cT = img.shape
    red=[]
    total=[]
    green=[]

    findobjects(outputs, img)    
    
    unique_data = [list(x) for x in set(tuple(x) for x in red)]

    RS = ['Risk Count','Safe Count']
    data = [len(unique_data), len(total)-len(unique_data)] 
    explode = (0.1, 0.3) 
    colors = ("Red","Green") 
    wp = { 'linewidth' : 1, 'edgecolor' : "Brown" } 
    fig, ax = plt.subplots(figsize =(10, 7))
    
    wedges, texts, autotexts=ax.pie(data, 
                                    autopct = lambda pct: func(pct, data), 
                                    explode = explode, 
                                    labels = RS, 
                                    shadow = True, 
                                    colors = colors, 
                                    startangle = 90, 
                                    wedgeprops = wp,
                                    textprops = dict(color ="black"))
    ax.legend(wedges, RS,title ="Count",loc ="center left", bbox_to_anchor =(1, 0)) 
    plt.setp(autotexts, size = 8, weight ="bold") 
    ax.set_title("Social Distancing Monitor") 

    plt.savefig('plot')
    pplot=cv2.imread('plot.png')

    
    
    cv2.putText(pplot,"Risk Count: {}".format(str(len(unique_data))),(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,250),2)
    cv2.putText(pplot,"Safe Count: {}".format(len(total)-len(unique_data)),(450,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,250,0),2)

    

    cv2.imshow('Social Distancing Monitor',pplot)
    cv2.imshow('Monitor',img)    
    
    plt.close()     #avoid memory leak
    os.remove('plot.png')
    if cv2.waitKey(1) & 0xFF ==ord('q'):
         break

