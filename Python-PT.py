# Import necessary libraries
from transformers import pipeline

# Load pre-trained sentiment-analysis model from Hugging Face
model = pipeline("sentiment-analysis")

# Define some sample movie reviews
reviews = [
    "I absolutely loved the movie. The storyline was fantastic!",
    "It was a terrible movie, the plot made no sense and the acting was horrible.",
    "The visuals were stunning, but the story could have been better.",
    "I wouldn't recommend this movie to anyone.",
    "The movie was okay, not the best but certainly not the worst."
]

# Use the pre-trained model to analyze the sentiment of the reviews
for review in reviews:
    result = model(review)
    print(f"Review: {review}\nSentiment: {result[0]['label']}, Confidence: {result[0]['score']:.2f}\n")
