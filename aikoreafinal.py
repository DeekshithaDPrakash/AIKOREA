import cv2
import numpy as np
import dlib
from math import hypot
import HandTrackingModule as htm
import time
from overlays import overlay_transparent

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
cap.set(cv2.CAP_PROP_FPS,30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

success, frame = cap.read()


bgArray = [
    cv2.imread("bgi/bgi00.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("bgi/bgi01.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("bgi/bgi02.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("bgi/bgi03.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("bgi/bgi04.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("bgi/bgi05.png", cv2.IMREAD_UNCHANGED)
]

for i in range(0, len(bgArray)):
    bgArray[i] = cv2.resize(bgArray[i], (1920,1080))

pTime=0

teeth1 = cv2.imread("img/Image for Dee/ff01.png", -1)
teeth2 = cv2.imread("img/Image for Dee/ff02.png", -1)
teeth3 = cv2.imread("img/Image for Dee/ff03.png", -1)
teeth4 = cv2.imread("img/Image for Dee/ff04.png", -1)

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (5).dat")

# loading hand detector
handdetector = htm.handDetector(detectionCon=0.8,  trackCon=0.8, maxHands=1000)

while True:
    success, frame = cap.read()
    cap.set(cv2.CAP_PROP_FPS,30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 9)
    frame.flags.writeable=False
    font = cv2.FONT_HERSHEY_SIMPLEX

    img = handdetector.findHands(frame, draw=True )
    lmList, bbox=handdetector.findPosition(img,draw=True)
    frame.flags.writeable=True
    faces = detector(frame)
    
    tipId=[4,8,12,16,20]
    if(len(lmList)!=0):
        fingers=[]
        if(lmList[tipId[0]][1]>lmList[tipId[0]-1][1]):
                fingers.append(1)
        else :
                fingers.append(0)
        #4 fingers
        for id in range(1,5):
            
          if(lmList[tipId[id]][2]<lmList[tipId[id]-2][2]):
                fingers.append(1)
            
          else :
                fingers.append(0)

    
        for face in faces:
            landmarks = predictor(frame, face)
            
            # Nose coordinates
            left_face = (landmarks.part(0).x, landmarks.part(0).y)
            right_face = (landmarks.part(16).x, landmarks.part(16).y)
            center_nose = (landmarks.part(31).x, landmarks.part(31).y)
            top_eyebrow = (landmarks.part(24).x, landmarks.part(24).y)
            bottom_chin = (landmarks.part(8).x, landmarks.part(8).y)

            width = int(hypot(left_face[0] - right_face[0],
                           left_face[1] - right_face[1])*2)
        
        

            height = int(hypot(top_eyebrow[0] - bottom_chin[0],
                           top_eyebrow[1] - bottom_chin[1])*2)
            top_left = (int(center_nose[0] - width / 2),
                              int(center_nose[1] - height / 1.5))
            if(fingers==[0,1,0,0,0] or fingers==[1,0,0,0,0]):
                # New nose position
                #top_left = (int(center_nose[0] - width / 2),
                              #int(center_nose[1] - height / 1.5))
                print(top_left)
        
                teethimage = cv2.resize(teeth1, (width, height),interpolation=cv2.INTER_CUBIC) 
                overlay_transparent(frame, teethimage, top_left[0], top_left[1])
              
                #imgFront = cv2.resize(bg1, (1920,1080))
                overlay_transparent(frame, bgArray[1], 0,0)
                
                
            elif(fingers==[0,1,1,0,0] or fingers==[1,1,0,0,0]):
                # New nose position
                #top_left = (int(center_nose[0] - width / 2),
                              #int(center_nose[1] - height / 1.5))
                teethimage = cv2.resize(teeth2, (width, height),interpolation=cv2.INTER_CUBIC)
                overlay_transparent(frame, teethimage, top_left[0], top_left[1])
             
                #imgFront = cv2.resize(bg2, (1920,1080))
                overlay_transparent(frame, bgArray[2], 0,0)
            
            elif(fingers==[0,1,1,1,0] or fingers==[1,1,1,0,0]):
                # New nose position
                #top_left = (int(center_nose[0] - width / 2),
                              #int(center_nose[1] - height / 1.5))
       
                teethimage = cv2.resize(teeth3, (width, height),interpolation=cv2.INTER_CUBIC)
                overlay_transparent(frame, teethimage, top_left[0], top_left[1])

                
                #imgFront = cv2.resize(bg3, (1920,1080))
                overlay_transparent(frame, bgArray[3], 0,0)

            elif(fingers==[0,1,1,1,1] or fingers==[1,1,1,1,0]):
                
                #top_left = (int(center_nose[0] - width / 2),
                              #int(center_nose[1] - height / 1.5))
      
                teethimage = cv2.resize(teeth4, (width, height),interpolation=cv2.INTER_CUBIC)
                overlay_transparent(frame, teethimage, top_left[0], top_left[1])

                #imgFront = cv2.resize(bg4, (1920,1080))
                overlay_transparent(frame, bgArray[4], 0,0)
            
            elif(fingers==[1,1,1,1,1]):
        
                #imgFront = cv2.resize(bg5, (1920,1080))
                overlay_transparent(frame, bgArray[5], 0,0)

    else:
        
        # imgFront = cv2.resize(bg0, (1920,1080))
        overlay_transparent(frame, bgArray[0], 0,0)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, f'FPS: {int(fps)}',(400,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),3)
    # cv2.namedWindow("image", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WND_PROP_FULLSCREEN)
    cv2.imshow("image", frame)  
    if cv2.waitKey(1) == ord('q'):
        break
                

            
