
import math
class Es:
    def __init__(self,EyePointsL,EyePointsR,FlmPredd):
        self.EyePointsR=EyePointsR
        self.EyePointsL=EyePointsL
        self.FlmPredd=FlmPredd

    def getWidth(EyePointsL,FlmPredd):
        LeftPoint=[FlmPredd.part(EyePointsL[0]).x,FlmPredd.part(EyePointsL[0]).y]
        RightPoint=[FlmPredd.part(EyePointsL[3]).x,FlmPredd.part(EyePointsL[3]).y]
        return (math.sqrt((RightPoint[0]-LeftPoint[0])**2+(RightPoint[1]-LeftPoint[1])**2))

    def getHeight(EyePointsL,FlmPredd):
        TopPtXY=[(FlmPredd.part(EyePointsL[1]).x+FlmPredd.part(EyePointsL[2]).x)/2,(FlmPredd.part(EyePointsL[1]).y+FlmPredd.part(EyePointsL[2]).y)/2]
        BtmPtXY=[(FlmPredd.part(EyePointsL[4]).x+FlmPredd.part(EyePointsL[5]).x)/2,(FlmPredd.part(EyePointsL[4]).y+FlmPredd.part(EyePointsL[5]).y)/2]
        return (math.sqrt((BtmPtXY[0]-TopPtXY[0])**2+(BtmPtXY[1]-TopPtXY[1])**2))

    def getEar(self):
        LeftHeight=self.getHeight(self.EyePointsL,self.FlmPredd)
        LeftWidth=self.getWidth(self.EyePointsL,self.FlmPredd)
        RghtHeight=self.getHeight(self.EyePointsR,self.FlmPredd)
        RghttWidth=self.getWidth(self.EyePointsR,self.FlmPredd)
        EarL= LeftWidth/LeftHeight
        EarR= RghttWidth/RghtHeight
        return (EarL+EarR)/2



#print(getHeight(EyePointsL,EyePointsR,lmPredd))
