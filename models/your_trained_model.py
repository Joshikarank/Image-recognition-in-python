import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Define data augmentation and preprocessing for training images
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

# Load training data
train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),  # Resize images to a common size
    batch_size=32,
    class_mode='categorical')

class_labels = train_generator.class_indices
sorted_class_labels = {k: v for k, v in sorted(class_labels.items())}
print(sorted_class_labels)
print(class_labels)

# Define the number of classes (car models)
num_classes = len(train_generator.class_indices)

# Create the model
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)

# Save the trained model
model.save('models/your_trained_model.h5')
