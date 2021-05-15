import dlib
import playsound
import GetEarMar

Threshold=0.5
CFrames=40
AoN=False
cntr=0
cntrA=0
AEyePointsL=[37,38,39,40,41,42]
AEyePointsR=[43,44,45,46,47,48]


def play():
    playsound('audio.mp3')
