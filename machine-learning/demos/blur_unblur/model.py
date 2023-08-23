import os
import shutil
import random
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Set up paths
source_dir = '/Users/tushargoyal/Documents/GitHub/mlpr'
blurred_dir = '/Users/tushargoyal/Documents/GitHub/mlpr'
train_dir = 'train_data'
test_dir = 'test_data'

# Create train and test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Move images and blurred images into corresponding train/test folders
for folder in ['images', 'blurred_images']:
    image_list = os.listdir(os.path.join(source_dir, folder))
    random.shuffle(image_list)
    
    train_size = int(0.8 * len(image_list))
    train_images = image_list[:train_size]
    test_images = image_list[train_size:]
    
    for image in train_images:
        src_path = os.path.join(source_dir, folder, image)
        dst_path = os.path.join(train_dir, folder, image)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)
    
    for image in test_images:
        src_path = os.path.join(source_dir, folder, image)
        dst_path = os.path.join(test_dir, folder, image)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data generators
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(128, 128), batch_size=32, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(128, 128), batch_size=32, class_mode='binary')

# Train the model
model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print("Test accuracy:", test_acc)

# Move final 20 images into a separate folder for testing
final_test_images = random.sample(os.listdir(os.path.join(source_dir, 'images')), 20)
final_test_folder = 'final_test_data'
os.makedirs(final_test_folder, exist_ok=True)
for image in final_test_images:
    src_path = os.path.join(source_dir, 'images', image)
    dst_path = os.path.join(final_test_folder, image)
    shutil.copy(src_path, dst_path)

# Test the model on final test images
final_test_datagen = ImageDataGenerator(rescale=1.0/255)
final_test_generator = final_test_datagen.flow_from_directory(final_test_folder, target_size=(128, 128), batch_size=32, class_mode=None, shuffle=False)
predictions = model.predict(final_test_generator)

for i, image in enumerate(final_test_images):
    prediction = predictions[i][0]
    if prediction >= 0.5:
        print(f"{image} is predicted as BLURRED with confidence {prediction:.2f}")
    else:
        print(f"{image} is predicted as NOT BLURRED with confidence {1 - prediction:.2f}")
