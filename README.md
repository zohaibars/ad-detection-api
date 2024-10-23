# ad-detection-api
Traning and infrence for ad dectection modules(Audio, video) using speechbrain and ResNet50



This repository contains a Ads Detection model capable of identifying specific Advertisement in audio and video recordings. It utilizes deep learning techniques to extract features and make predictions based on voice/frame samples.

Installation
1- Clone the repository:                             git clone https://github.com/AI-TEAM-R-D-Models/ad-detection-api.git
2- Navigate into the cloned directory:               cd ad-detection-api
3- Install the required dependencies:                pip install -r req.txt



Training

To train the model, follow these steps:

1- Organize your training data and Training/Detecting:

            * For Ads Detection using Audio
            
            Run the Command python trainapi.py
            It will start the Training API and in that API you have to give the Advertisement you want to detect with the time, the time specify that for how long you will be able to detect this ad.
            
            
            After Training Run the command python predictapi.py
            It will start the Prediction API and in that API you have to provide mp3/wav file in which you want to detect the trained ad.
            
            *For Ads Detection using Frame
            
            Run the Command python TrainGPU.py
            It will start the Training API and in that API you have to give the Advertisement(video file) you want to detect with the time, the time specify that for how long you will be able to detect this ad.
            
            
            After Training Run the command python PredictGPU.py
            It will start the Prediction API and in that API you have to provide video file in which you want to detect the trained ad.









