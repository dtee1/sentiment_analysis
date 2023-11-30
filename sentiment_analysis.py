import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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

train_reviews, train_labels = read_reviews('aclImdb_v1/aclImdb/train')
test_reviews, test_labels = read_reviews('aclImdb_v1/aclImdb/test')

# Function to load GloVe embeddings
def load_glove_embeddings(embeddings_file):
    embeddings_index = {}
    with open(embeddings_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    return embeddings_index

# Load GloVe embeddings
glove_file = 'glove.6B/glove.6B.100d.txt'
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

# Build and compile the RNN model
def build_rnn_model(units=128, learning_rate=1e-3):
    model = tf.keras.Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sequence_length, trainable=False))
    model.add(SimpleRNN(units=units, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
X_train = np.array(X_train)
X_valid = np.array(X_valid)

# Build and compile the model
model = build_rnn_model()

# Train the model on the entire training set
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_valid, y_valid))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")