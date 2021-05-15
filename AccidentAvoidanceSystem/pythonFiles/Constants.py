import dlib
import playsound
import GetEarMar

Threshold=0.5   #Threshold for ear
CFrames=40      # No of frames for which threshold is .5
AoN=False
cntr=0
cntrA=0
AEyePointsL=[37,38,39,40,41,42]   #Left Eye Landmarks
AEyePointsR=[43,44,45,46,47,48]    #Right eye Landmarks 


def play():
    playsound('audio.mp3')
