import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torch.autograd import Variable

import numpy as np
from PIL import Image

from app.config import INFERENCE_ON_CPU

class Image_Gender():
    def __init__(self, modelPath):

        self.modelPath = modelPath

        self.genderList = ['female', 'male']
        self.n_genders = 2

        if INFERENCE_ON_CPU:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(self.device)
        self.res_gender = self.get_res18_for_eval(self.modelPath, self.n_genders)
        _=self.res_gender.to(self.device)     

    def get_res18_for_eval(self,modelPath, n_classes):
        """Gets the trained model."""
        
        print("Creating Resnet Gender...")
        model = torchvision.models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,n_classes)

        print("Loading Weights...")  
        model.load_state_dict(torch.load(modelPath, map_location=self.device))
        model.eval()
        return model


    def get_transform(self):

        transform = transforms.Compose([
            transforms.Resize(205),
            transforms.CenterCrop(200),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        return transform

    def predict_class_from_image(self,image, model, classList):
        
        transform = self.get_transform()
        image = transform(image).float()
        image = Variable(image, requires_grad=False)
        image = image.unsqueeze(0)

        with torch.no_grad():
            outputs = model(image.to(self.device))
        
        prob = list(torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu().numpy())
        prob = [str(p) for p in prob]
        result = dict(zip(classList,prob))

        return result

    def predict_gender_image(self,image):
        return self.predict_class_from_image(image, self.res_gender, self.genderList)
