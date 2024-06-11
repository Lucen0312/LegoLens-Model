import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the saved model
model = tf.keras.models.load_model('best_model.h5')

# Create a data generator for the test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'data/test',  # Replace with your actual test data directory
    target_size=(224, 224),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_generator)

print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')