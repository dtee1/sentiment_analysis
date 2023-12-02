from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Function to generate word cloud
def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

positive_reviews = [review['text'] for review in train_data[0] if review['rating'] > 6]
positive_reviews = [' '.join(review) for review in positive_reviews]  # Ensure each review is a string
positive_text = ' '.join(positive_reviews)
generate_word_cloud(positive_text, 'Word Cloud for Positive Reviews')

negative_reviews = [review['text'] for review in train_data[0] if review['rating'] < 5]
negative_reviews = [' '.join(review) for review in negative_reviews]  # Ensure each review is a string
negative_text = ' '.join(negative_reviews)
generate_word_cloud(negative_text, 'Word Cloud for Negative Reviews')


import matplotlib.pyplot as plt
import seaborn as sns

# Function to generate scatter plot
def plot_review_length_vs_rating(data, title):
    review_lengths = [len(review['text']) for review in data]
    ratings = [review['rating'] for review in data]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=review_lengths, y=ratings)
    plt.title(title)
    plt.xlabel('Review Length')
    plt.ylabel('Rating')
    plt.show()

# Generate scatter plot for train data
plot_review_length_vs_rating(train_data[0], 'Review Length vs. Rating (Train Data)')

# Generate scatter plot for test data
plot_review_length_vs_rating(test_data[0], 'Review Length vs. Rating (Test Data)')

def extract_ratings(data):
    ratings = [review['rating'] for review in data]
    return ratings

train_ratings_plot = extract_ratings(train_data[0])

test_ratings_plot = extract_ratings(test_data[0])

print("Train Ratings:", train_ratings_plot)
print("Test Ratings:", test_ratings_plot)

import matplotlib.pyplot as plt

def plot_ratings_distribution(ratings, dataset_name):
  plt.hist(ratings, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], align='mid', rwidth=0.8, color='skyblue', edgecolor='black')
  plt.xlabel('Rating')
  plt.ylabel('Number of Reviews')
  plt.title(f'Distribution of Ratings - {dataset_name}')
  plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  plt.show()

plot_ratings_distribution(train_ratings_plot, 'Train Data')

# Plot distribution for test data
plot_ratings_distribution(test_ratings_plot, 'Test Data')