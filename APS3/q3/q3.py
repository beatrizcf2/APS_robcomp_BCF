#!/usr/bin/python
# -*- coding: utf-8 -*-


# Para RODAR
# python object_detection_webcam.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
# Credits: https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

print("Para executar:\npython q3.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel")

# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)


def detect(frame):
    global dici
    image = frame.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    results = []

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence


        if confidence > args["confidence"]:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # display the prediction if appears in more than 5 frames in a roll
            if CLASSES[idx]=='person' and CLASSES[idx] in dici:
                if dici[CLASSES[idx]] > 5 :
                    aparece = "{0} aparece {1} vezes".format(CLASSES[idx],dici[CLASSES[idx]])
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    print("[INFO] {}".format(label))
                    cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                    cv2.putText(image, aparece, (50,60), cv2.FONT_HERSHEY_SIMPLEX ,1,(0,50,100),2,cv2.LINE_AA)

            results.append((CLASSES[idx], confidence*100, (startX, startY),(endX, endY) ))

    # show the output image
    return image, results






import cv2

#cap = cv2.VideoCapture('hall_box_battery_1024.mp4')
cap = cv2.VideoCapture(0)

print("Known classes")
print(CLASSES)
dici = {}




while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    lista = []
    
    result_frame, result_tuples = detect(frame)
    print("lista", lista)
    #para cada index detectado no frame eu add numa lista
    for t in result_tuples:
        print(result_tuples)
        lista.append(t[0])
        #se o index nao tiver no dici eu add
        if t[0]=='person':
            if t[0] not in dici:
                contador = 1
                dici[t[0]] = contador
            #se nao so altera o contador
            else:
                dici[t[0]] +=1

    #se o que ta no dici nao tiver na lista do que foi detectado, deleta do dici
    
    
    if 'person' not in lista and 'person' in dici:
        del dici['person']
    
    '''
    print("lista2", lista,'dici',dici)
    for i in lista:
        if i not in dici:
            dici[i]=0
            
    '''


    # Display the resulting frame
    cv2.imshow('frame',result_frame)

    # Prints the structures results:
    # Format:
    # ("CLASS", confidence, (x1, y1, x2, y3))
    for t in result_tuples:
        print(t[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
