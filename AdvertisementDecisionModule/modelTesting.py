import pandas as pd
import joblib

# Load the saved model pipeline and label encoder
pipeline = joblib.load('model_pipeline.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Create a new sample DataFrame for testing
# This sample uses only one keyword ("organic") to test with partial keywords.
new_sample = pd.DataFrame({
    "age": [34],
    "gender": ["F"],
    "keywords": ["daily"]
})

# Predict the category for the new sample
predicted_category_encoded = pipeline.predict(new_sample)
predicted_category = label_encoder.inverse_transform(predicted_category_encoded)

print("Predicted Category with partial keywords:", predicted_category[0])
