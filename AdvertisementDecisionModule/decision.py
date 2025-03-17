import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import joblib

# Load the CSV file into a DataFrame
df = pd.read_csv("data.csv")

# Our features: age, gender, keywords
# Target: category (the advertisement category)
X = df[["age", "gender", "keywords"]]
y = df["category"]

# Encode target labels (categories) as numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Create a column transformer to process each feature type appropriately:
# - 'age' is scaled,
# - 'gender' is one-hot encoded,
# - 'keywords' is vectorized with TF-IDF.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["age"]),
        ("cat", OneHotEncoder(), ["gender"]),
        ("text", TfidfVectorizer(), "keywords")
    ]
)

# Build a pipeline that first transforms the data then trains a classifier.
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train the classifier
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Print evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# persist/save the model so I can load it later
joblib.dump(pipeline, 'model_pipeline.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

