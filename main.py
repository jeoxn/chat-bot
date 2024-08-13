import json
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load JSON data
with open('intents.json') as file:
    data = json.load(file)

# Initialize lists for sentences and corresponding tags
sentences = []
tags = []

# Extract patterns and tags from the data
for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        tags.append(intent['tag'])

# Encode the tags
label_encoder = LabelEncoder()
encoded_tags = label_encoder.fit_transform(tags)

# Convert sentences to TF-IDF feature vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

# Create and train the RF model
model = RandomForestClassifier()
model.fit(X, encoded_tags)

# Function to classify user input and generate a response
def classify_and_respond(user_input):
    # Preprocess the input
    user_input_vector = vectorizer.transform([user_input])

    # Predict the intent
    prediction = model.predict(user_input_vector)
    predicted_tag = label_encoder.inverse_transform(prediction)[0]

    # Select a random response
    for intent in data['intents']:
        if intent['tag'] == predicted_tag:
            response = random.choice(intent['responses'])
            break

    return response

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    response = classify_and_respond(user_input)
    print("Bot:", response)
