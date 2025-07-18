import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load flowers dataset and information
dataset, info = tfds.load('tf_flowers', with_info=True, as_supervised=True)
full_dataset = dataset['train']

# Training size is 70% of the dataset
total = info.splits['train'].num_examples
train_size = int(0.7 * total)
val_size = int(0.15 * total)

# Train and test sets
full_dataset = full_dataset.shuffle(total, seed=42)
train = full_dataset.take(train_size)
validation = full_dataset.skip(train_size).take(val_size)
test = full_dataset.skip(train_size + val_size)

num_classes = info.features['label'].num_classes

# Resizing the image to 150, 150 and normalizing
def preprocess(image, label):
    image = tf.image.resize(image, [150, 150])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train = train.map(preprocess)
validation = validation.map(preprocess)
test = test.map(preprocess)

# Transforming the data
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
    layers.RandomContrast(0.2)
])

SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE
BATCH_SIZE = 32

# Final training and testing datasets with parameters
train_dataset = (train 
                 .map(lambda x, y: (data_augmentation(x, training=True), y))
                 .cache()
                 .shuffle(SHUFFLE_BUFFER_SIZE)
                 .batch(BATCH_SIZE)
                 .prefetch(PREFETCH_BUFFER_SIZE))

val_dataset = (validation
               .cache()
               .batch(BATCH_SIZE)
               .prefetch(PREFETCH_BUFFER_SIZE))

test_dataset = (test
                .cache()
                .batch(BATCH_SIZE)
                .prefetch(PREFETCH_BUFFER_SIZE))

# Building the model with convolutions and max pooling
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Callback method. Stops at 90% accuracy
class StoppingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') >= 0.9):
            self.model.stop_training = True

callbacks = StoppingCallback()

# Compiling the model with a loss, optimizer, and metric
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.01),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Model history for the training dataset with 15 epochs and the callback
history = model.fit(
    train_dataset,
    validation_data = val_dataset,
    epochs=15,
    verbose=2,
    callbacks=[callbacks]
)

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']


# Plotting the accuracy
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluating on the test set
model.evaluate(test_dataset)

model.save('flower_classifier_model')
