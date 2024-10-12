import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Sample dataset: Movie reviews with labels (positive/negative)
data = {
    'Review': [
        'I loved the movie', 'The movie was awful', 'Great story and acting',
        'Worst movie ever', 'I enjoyed the film', 'The plot was boring', 'Fantastic movie'
    ],
    'Sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative','positive']
}

df = pd.DataFrame(data)

# Encode labels
label_encoder = LabelEncoder()
df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Sentiment'], test_size=0.2, random_state=42)

# Build a simple model
model = keras.Sequential([
    layers.Embedding(input_dim=5000, output_dim=16, input_length=50),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Convert text data to numerical data
tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=50)
X_test_pad = keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=50)

# Train the model
model.fit(X_train_pad, y_train, epochs=5, batch_size=4, validation_data=(X_test_pad, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f'Accuracy: {accuracy*100:.2f}%')

# Test with a new review
new_review = ['The movie was fantastic!']
new_review_seq = tokenizer.texts_to_sequences(new_review)
new_review_pad = keras.preprocessing.sequence.pad_sequences(new_review_seq, maxlen=50)

prediction = model.predict(new_review_pad)
sentiment = 'positive' if prediction[0] > 0.5 else 'negative'
print(f'Sentiment: {sentiment}')
