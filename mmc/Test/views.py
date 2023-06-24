
from django.conf import settings
from django.shortcuts import render, HttpResponse
import json
import sys
from templates.prediction import prediction1d, prediction2d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
import os

# Create your views here.
# Audio Upload
def audioUpload(request):
    return render(request, 'AudioUpload.html')

def toModelSelection(request):
    file = request.GET['file']
    mood = request.GET['mood']
    fnm = {'file': file, 'mood': mood}
    return render(request, 'ModelSelection.html', {'fnm': json.dumps(fnm)})

def getBack(request):
    return render(request, 'AudioUpload.html')

def toPrediction(request):
    file = request.GET['file']
    mood = request.GET['mood']
    model = request.GET['model']
    types = ['angry','happy','relaxed','sad']  # For plotting the x-axis
    if(os.path.isfile("./static/image/prob.png")):
        os.remove("./static/image/prob.png")
    if(os.path.isfile("./static/image/prob1.png")):
        os.remove("./static/image/prob1.png")
    if(os.path.isfile("./static/image/prob2.png")):
        os.remove("./static/image/prob2.png")
    if(os.path.isfile("./static/image/prob3.png")):
        os.remove("./static/image/prob3.png")
    
    if model =='1':
        predict,prob = prediction1d('templates/'+file)
        f=plt.figure(figsize=(12,4))
        plt.title('probabilty of being different mood types')
        plt.ylabel('probabilty')
        plt.bar(types, prob[0])
        plt.savefig('./static/image/prob.png')
        
    else:
        predict,prob1,prob2,prob3 = prediction2d('templates/'+file)
        
        f = plt.figure(figsize=(12,4))
        plt.title('probabilty of being different mood types using spectrogram')
        plt.bar(types, prob1[0])
        plt.savefig('./static/image/prob1.png')

        f = plt.figure(figsize=(12,4))
        plt.title('probabilty of being different mood types using MFCC')
        plt.bar(types, prob2[0])
        plt.savefig('./static/image/prob2.png')
        
        f = plt.figure(figsize=(12,4))
        plt.title('probabilty of being different mood types using Mel-spectrogram')
        plt.bar(types, prob3[0])
        plt.savefig('./static/image/prob3.png')
        

    fmm = {'file': file, 'mood': mood, 'model': model,'predict':predict}
  
    return render(request, 'Prediction.html', {'fmm': json.dumps(fmm)})

def returnModelSelection(request):
    file = request.GET['file']
    mood = request.GET['mood']
    fnm = {'file':file,'mood':mood}
    render(request, 'ModelSelection.html', {'fnm': json.dumps(fnm)})


    










