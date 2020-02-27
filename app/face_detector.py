from PIL import Image
import dlib
import numpy as np


class Face_Detector():
    def __init__(self):

        self.dlib_det = dlib.get_frontal_face_detector()

    def getFace(self,frame):
        dfaces = self.dlib_det(np.array(frame),2)
        if len(dfaces)>0:
                d = dfaces[0]
                croppedFrame = frame.crop((d.left(), d.top(), d.right(), d.bottom()))
                return croppedFrame
        else:
            return None