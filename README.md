# INF2009_Team22

<h1>
Added a new folder called 'AdvertisementDecisionModule'
</h1>

<h3>The AdvertisementDecisionModule contains:</h3>

- `dummyDataGenerator.py` -> run to create the dummy data file.
- `decision.py` -> run to create the model, which is then persisted/saved into 2 `.pkl` files.
- `modelTesting.py` -> run to test the model, change the age, gender and keywords. It returns a category.

<h3>The pickle files:</h3>

- `model_pipeline.pkl` -> stores the pipeline, including preprocessing(scaling, one-hot encoding, TF-IDF transformation) and the classifier. Basically transforms the raw inputs into the proper format.
- `label_encoder.pkl` -> maps the textual advertisement categories to numeric labels during training. When you make predictions, the model outputs numeric labels. You then use the label encoder to convert these numeric predictions back into their original category names.

Webserver link: [https://github.com/ivanpyw1999/edge_inf2009_22](https://github.com/ivanpyw1999/edge_inf2009_22)

<h1> Design Implementations </h1>

<h3>SpeechAnalysisModule</h3>
- Uses Silero_VAD to detect speech from potentially noisy background noise
- Uses Vosk API small en model for keyword spotting
  
  - Model adaptation is done to boost probabilities of keywords. See our 
    [model adaptation notebook](https://github.com/ImUrGit/INF2009_Team22/blob/main/SpeechAnalysisModule/Vosk_API_Model_Graph_Adaptation.ipynb)

<h3>ImageBasedAnalysisModule</h3>
- Uses Deepface model to detect age and gender of user
    
  - Quantization is used to increase inference speed.

<h3>AdvertisementDecisionModule</h3>
- Uses a linear regression to predict user's product category preferences based on their age, gender, and/or spoken keywords.
