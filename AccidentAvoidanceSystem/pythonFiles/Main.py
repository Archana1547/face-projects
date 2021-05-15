import dlib
import imutils
import cv2,time
import numpy as np
import Constants as c
import GetEarMar as p
from threading import Thread
import math
import playsound

t1=[]

                        
   
 #function to stop music 
def stop(t):
    while True:
        global stopT
        if stopT:
            break
            
     #function to play music 
def play():
    playsound.playsound('alarm.mp3')

 #function to get width of Left & Right Eye Points 
def getWidth(EyePointsL,FlmPredd):
    LeftPoint=[FlmPredd.part(EyePointsL[0]).x,FlmPredd.part(EyePointsL[0]).y]
    RightPoint=[FlmPredd.part(EyePointsL[3]).x,FlmPredd.part(EyePointsL[3]).y]
    return (math.sqrt((RightPoint[0]-LeftPoint[0])**2+(RightPoint[1]-LeftPoint[1])**2))

   #function to get height of Left & Right Eye Points 
def getHeight(EyePointsL,FlmPredd):
    TopPtXY=[(FlmPredd.part(EyePointsL[1]).x+FlmPredd.part(EyePointsL[2]).x)/2,(FlmPredd.part(EyePointsL[1]).y+FlmPredd.part(EyePointsL[2]).y)/2]
    BtmPtXY=[(FlmPredd.part(EyePointsL[4]).x+FlmPredd.part(EyePointsL[5]).x)/2,(FlmPredd.part(EyePointsL[4]).y+FlmPredd.part(EyePointsL[5]).y)/2]
    return (math.sqrt((BtmPtXY[0]-TopPtXY[0])**2+(BtmPtXY[1]-TopPtXY[1])**2))

   #function to get ratio of Eye Points 
def getEar(EyePointsL,FlmPredd,EyePointsR):
    LeftHeight=getHeight(EyePointsL,FlmPredd)
    LeftWidth=getWidth(EyePointsL,FlmPredd)
    RghtHeight=getHeight(EyePointsR,FlmPredd)
    RghttWidth=getWidth(EyePointsR,FlmPredd)
    EarL= LeftWidth/LeftHeight
    EarR= RghttWidth/RghtHeight
    return (EarL+EarR)/2

  #Detecting face in a live video 
detector=dlib.get_frontal_face_detector()
#e=p.Es(c.AEyePointsL,c.AEyePointsR)
lm_file="shape_predictor_68_face_landmarks.dat"
lm_predictor=dlib.shape_predictor(lm_file)

WebCam=cv2.VideoCapture(0)
frame=1
while (1):
    frame+=1
    flag,video=WebCam.read()
    video = cv2.resize(video, (0, 0), fx=0.25, fy=0.25)
    gray=cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
    rects=detector(gray,0)
    for i in rects:
        lmPredd=lm_predictor(gray,i)
        for i in range(0,68):
            x=lmPredd.part(i).x
            y=lmPredd.part(i).y
            #print("Detecting x & y No: ",i)
            cv2.circle(video, (x, y), 1, (80, 58, 255), -1)
            ear=getEar(c.AEyePointsL,lmPredd,c.AEyePointsR)
            #print(ear)
            if(ear<c.Threshold):
                c.cntr+=1
                if c.cntr >= c.CFrames:
                    if not c.AoN:
                        t = Thread(target=play)
                        t.deamon = True
                        t.start()
                        stopT=True
                        #stop(t)
                        #print("Thread Killed")
        
                        #c.cntr=0
                    cv2.putText(video, "DROWSINESS ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
            else:
                c.cntr=0
                AoN=False
               # print("In else loop ",c.cntr)
    cv2.imshow("Facial landmark",video)

    key=cv2.waitKey(1)
    if key==ord('q'):
        break
WebCam.release()
cv2.destroyAllWindows()
