import json


import torch
import torch.nn as nn
import unicodedata
import string

from app.CharRnn import CharRNN as RNN



class Name_Gender():
    def __init__(self, nameDictFile, gender_model_path):

        self.nameDictFile = nameDictFile
        with open(self.nameDictFile, 'r') as outfile:
            self.nameDict = json.load(outfile)
        
        self.all_letters = string.ascii_letters + " .,;'"
        self.n_letters = len(self.all_letters)
        self.n_hidden = 128
        self.n_categories = 2


        self.gender_model_path = gender_model_path
        self.gender_model = self.get_gender_model_for_eval(self.gender_model_path)
        self.all_genders = ['male','female']

    def lookup_dict(self, name):
        if name in self.nameDict:
            cnt = self.nameDict[name]['M'] + self.nameDict[name]['F']
            m_p = self.nameDict[name]['M'] / cnt
            f_p = self.nameDict[name]['F'] / cnt
            return {'male':str(m_p), 'female':str(f_p)}
        else:
            return None

    def get_gender_model_for_eval(self,modelPath):
        """Gets the trained model."""
        
        print("Creating RNN Gender...")
        model = RNN(self.n_letters, self.n_hidden, self.n_categories)
        
        print("Loading Weights...")  
        model.load_state_dict(torch.load(modelPath))
        model.eval()
        return model

    def letterToIndex(self,letter):
        return self.all_letters.find(letter)

    def letterToTensor(self,letter):
        tensor = torch.zeros(1, self.n_letters)
        tensor[0][self.letterToIndex(letter)] = 1
        return tensor

    def lineToTensor(self,line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor


    def evaluate(self,line_tensor, model):
        hidden = model.initHidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = model(line_tensor[i], hidden)

        return output


    def predict(self,input_line, all_categories, model, n_predictions=2):
    #     print('\n> %s' % input_line)
        with torch.no_grad():
            output = self.evaluate(self.lineToTensor(input_line), model)

            # Get top N categories
            topv, topi = output.topk(n_predictions, 1, True)
            topv = torch.nn.functional.softmax(topv, dim=1)
            predictions = {}

            for i in range(n_predictions):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                predictions[all_categories[category_index]] = str(value)
        return predictions


    def predict_gender(self, name, lookup =True):
        name = name.lower()
        if lookup:
            lookup_status = self.lookup_dict(name)
            if lookup_status:
                return lookup_status
        return self.predict(name, self.all_genders, self.gender_model)
