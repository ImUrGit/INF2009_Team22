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