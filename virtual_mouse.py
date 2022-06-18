import cv2
import time
import numpy as np 
from cvzone.HandTrackingModule import HandDetector
import autopy

wCam , hCam = 640 , 480
wScr , hScr = autopy.screen.size()
frameR = 100

smoothening = 20
plocX,plocY = 0,0
clocX,clocY = 0,0

cap = cv2.VideoCapture(0)
cap.set(3,hCam)
cap.set(4,wCam)
pTime = 0
detector = HandDetector(detectionCon=0.8,maxHands=1)

while True:
    #1.Find hand landmarks
    success, img = cap.read()
    img = cv2.flip(img , 1)
    hands , img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList']
        bbox = hands[0]['bbox']

        #2.Get the tip of the index and middle fingers
        if len(lmList)!=0:
            x1 = lmList[8][0] #Index finger pos
            y1 = lmList[8][1]

            x2 = lmList[12][0] #Middle finger pos
            y2 = lmList[12][1]

            #3.Check which fingers are up
            fingers = detector.fingersUp(hands[0])
            #print(fingers)
            cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)

            #4.Only index finger:moving mode
            if fingers[1]==1 and fingers[2]==0:
                #5.Convert coordinates
                
                x3 = np.interp(x1, (frameR,wCam-frameR), (0,wScr))
                y3 = np.interp(y1, (frameR,hCam-frameR), (0,hScr))

                #6.Smoothen values
                clocX = plocX +(x3-plocX)/smoothening
                clocY = plocY +(y3-plocY)/smoothening

                #7.Move mouse
                autopy.mouse.move(clocX, clocY)
                cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
                plocX,plocY = clocX,clocY

            #8.Both index and middle fingers are up:Clicking mode
            if fingers[1]==1 and fingers[2]==1:
                #9.Find distance between fingers
                length,lineInfo,img = detector.findDistance((x1,y1),(x2,y2),img)
                #print(length)
                #10.Click mouse if distance short
                if length < 40:
                    cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(0,255,0),cv2.FILLED)
                    autopy.mouse.click()   
    
    #11.Frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0),3)


    #12.Display
    cv2.imshow("Image",img)
    cv2.waitKey(1)

