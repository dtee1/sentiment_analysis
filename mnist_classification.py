import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt

def train_and_evaluate_on_gpu():
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Build a simple convolutional neural network (CNN) model
    model = models.Sequential([
        layers.Reshape((28, 28, 1), input_shape=(28, 28)),  # Add a channel dimension
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 output classes for digits 0-9
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Display the model summary
    model.summary()

    # Train the model on the training data
    with tf.device('/device:GPU:0'):
        history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc}')

    # Plot the training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    train_and_evaluate_on_gpu()