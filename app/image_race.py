import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torch.autograd import Variable

import numpy as np
from PIL import Image

from app.config import INFERENCE_ON_CPU

class Image_Race():
    def __init__(self, modelPath):

        self.modelPath = modelPath

        self.race_classes = ['Asian', 'Black', 'South-Asian', 'Others', 'White']
        self.n_race = 5


        if INFERENCE_ON_CPU:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.res_race = self.get_res18_for_eval(self.modelPath, self.n_race)
        print(self.device)
        _=self.res_race.to(self.device)     

    def get_res18_for_eval(self,modelPath, n_classes):
        """Gets the trained model."""
        
        print("Creating Resnet Race...")
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
        result = dict(zip(classList,prob))

        return result

    
    def predict_race_image(self, image, topk=3):
        result = self.predict_class_from_image(image, self.res_race, self.race_classes)
        result = {k: str(v) for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)[:topk]}
        return result