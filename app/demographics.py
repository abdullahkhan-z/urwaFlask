
import os
import json
import datetime
import numpy as np

from PIL import Image


from app.name_gender import Name_Gender
from app.image_gender import Image_Gender
from app.image_age import Image_Age
from app.image_race import Image_Race
from app.face_detector import Face_Detector


from app.utils import get_PIL_image_url
from app.config import *



class Demographics():
    def __init__(self):

        self.name_gender_predictor = Name_Gender(NAME_GENDER_DICT_PATH,NAME_GENDER_MODEL_PATH)

        self.image_gender_predictor = Image_Gender(IMAGE_GENDER_MODEL_PATH)

        self.image_age_predictor = Image_Age(IMAGE_AGE_MODEL_PATH)

        self.image_race_predictor = Image_Race(IMAGE_RACE_MODEL_PATH)

        self.face_detect = Face_Detector()


    def get_face(self, prof):

        if 'image' in prof:
            try:
                image = get_PIL_image_url(prof['image'])
                face = self.face_detect.getFace(image)
                if face:
                    return True, face
                else:
                    return False,None
            except:
                return False,None

        else: 
            return False,None


    def get_first_name(self, prof):

        if 'name' in prof:
            try:
                firstName = prof['name'].split(' ')[0]
                if len(firstName) > 1:
                    return True,firstName
                else:
                    return False,None
            except:
                return False,None
                
        else:
            return False,None
        
    def get_graduation_year(self, prof):

        if 'graduation' in prof:
            try:
                graduationYear = int(prof['graduation'][:4])
                return True, graduationYear
            except:
                return False,None
                
        else:
            return False,None
    

    def predict_age_graduation(self, graduationYear):
        current_year = datetime.datetime.now().year
        avg_grad_age = 22
        return current_year - graduationYear + avg_grad_age


    def get_demographics(self,prof):
        
        result = {}

        validImage, face = self.get_face(prof)
        validName, firstName = self.get_first_name(prof)
        validGradYear, grad_year = self.get_graduation_year( prof)

        # gender
        if validImage:
            image_gender = self.image_gender_predictor.predict_gender_image(face)
            result['gender'] = image_gender
        
        elif validName:
            gender_name = self.name_gender_predictor.predict_gender(firstName)
            result['gender'] = gender_name            
        

        #age
        if validGradYear:
            grad_age = self.predict_age_graduation(grad_year)
            result['age'] = grad_age
        
        elif validImage:
            image_age = self.image_age_predictor.predict_age_image(face)
            result['age'] = image_age

        # race
        if validImage:

            image_race = self.image_race_predictor.predict_race_image(face)
            result['race'] = image_race
        
        
        return result




    

