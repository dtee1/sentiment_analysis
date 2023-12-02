import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
# Read the reviews, extract unique id, text, and rating
def read_reviews(directory):
    reviews = []
    labels = []
    stop_words = set(stopwords.words('english'))

    for label in ['pos', 'neg']:
        path = os.path.join(directory, label)
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                unique_id, rating = filename[:-4].split('_')
                review = file.read()

                tokens = word_tokenize(review)
                tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

                reviews.append({'id': int(unique_id), 'text': tokens, 'rating': int(rating)})
                labels.append(1 if label == 'pos' else 0)
    return reviews, labels

train_reviews, train_labels = read_reviews('aclImdb/train')
test_reviews, test_labels = read_reviews('aclImdb/test')

def load_glove_embeddings(embeddings_file):
    embeddings_index = {}
    with open(embeddings_file, encoding='utf-8') as f:
        for line in f:
            # Skip header line if present
            if line.startswith(' '):
                continue

            # Skip empty lines
            if not line.strip():
                continue

            values = line.split()
            word = values[0]

            # Check if any element in values[1:] is not a valid float
            try:
                vector = np.asarray(values[1:], dtype='float32')
            except ValueError:
                continue

            embeddings_index[word] = vector

    return embeddings_index

# Load GloVe embeddings using emmbed_dict
glove_file = 'glove.840B.300d.txt'
glove_embeddings = load_glove_embeddings(glove_file)

# Build vocabulary using GloVe embeddings
word_to_index = {}
index = 1
for word in glove_embeddings.keys():
    word_to_index[word] = index
    index += 1
    
# Tokenize and pad sequences
max_sequence_length = 200
vocab_size = len(word_to_index) + 1
train_sequences = [[word_to_index.get(word, 0) for word in review['text']] for review in train_reviews]
train_data = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_sequences = [[word_to_index.get(word, 0) for word in review['text']] for review in test_reviews]
test_data = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Create the model with GloVe embeddings
embedding_dim = 100  # Assuming you are using the 100-dimensional GloVe embeddings

model = tf.keras.Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length, trainable=False,
              weights=[np.array([glove_embeddings.get(word, np.zeros(embedding_dim)) for word in word_to_index.keys()])]),
    SimpleRNN(units=64),
    Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print a summary of the dataset
print("Summary of the IMDb Dataset:")
print("Number of training samples:", len(X_train))
print("Number of validation samples:", len(X_valid))
print("Number of testing samples:", len(test_data))
print("Vocabulary Size:", vocab_size)
print("Max Sequence Length:", max_sequence_length)
print("Embedding Dimension:", embedding_dim)
print("\nExample of a training sample:")
print(X_train[0])
print("Corresponding sentiment label for the example:", y_train[0])
print("\nModel Summary:")
model.summary()