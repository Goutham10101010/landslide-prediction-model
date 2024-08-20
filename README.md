# landslide-prediction-model
model to prediction model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
from tkinter import Tk, filedialog
from shutil import copyfile

# Function to select and save image
def select_and_save_image():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", ".jpg;.jpeg;*.png")])
    if file_path:
        desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        new_folder_path = os.path.join(desktop_path, 'Selected_Images')
        os.makedirs(new_folder_path, exist_ok=True)
        new_file_path = os.path.join(new_folder_path, os.path.basename(file_path))
        copyfile(file_path, new_file_path)
        print(f"Image saved to {new_file_path}")
        return new_file_path
    else:
        print("No file selected")
        return None

# Select and save the image
image_path = select_and_save_image()

# Check if image_path is None before proceeding
if image_path is not None:
    # Data Preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Ensure the directories exist
    train_dir = train folder(add path to train)
    validation_dir = validation folder(add path to validation model)
    
    if not os.path.exists(train_dir) or not os.path.exists(validation_dir):
        print("Error: Training or validation directory not found.")
    else:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary'
        )

        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary'
        )

        # Debugging output
        print(f"Number of training samples: {train_generator.samples}")
        print(f"Number of validation samples: {validation_generator.samples}")

        # Ensure there are samples to train on
        if train_generator.samples == 0 or validation_generator.samples == 0:
            print("Error: No images found in training or validation directory.")
        else:
            # Building the Model
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(512, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            # Compiling the Model
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Training the Model
            history = model.fit(
                train_generator,
                steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
                epochs=15,
                validation_data=validation_generator,
                validation_steps=max(1, validation_generator.samples // validation_generator.batch_size)
            )

            # Save the Model
            model_save_path = os.path.join(desktop_path, 'landslide_predictor.h5')
            model.save(model_save_path)
            print(f"Model saved to {model_save_path}")
