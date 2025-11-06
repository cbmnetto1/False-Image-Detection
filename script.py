import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Config
class DatasetConfig:
    DATA_PATH = 'dataset'
    TRAIN_CSV = os.path.join(DATA_PATH, 'train.csv')
    TEST_CSV = os.path.join(DATA_PATH, 'test.csv')
    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 10
    MODEL_FILE = 'AI-vs-Human-Classifier.h5'


# Load and Prepare Data
train_df = pd.read_csv(DatasetConfig.TRAIN_CSV)
test_df = pd.read_csv(DatasetConfig.TEST_CSV)

train_df["file_name"] = train_df["file_name"].apply(lambda x: os.path.join(DatasetConfig.DATA_PATH, x))
train_df = train_df.dropna(subset=["file_name", "label"])
train_df = train_df[train_df["file_name"].apply(os.path.exists)]
train_df["label"] = train_df["label"].astype(str).apply(lambda x: "human" if x.strip() == "0" else "ai")

# Data Generators
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.1
)

train_generator = datagen.flow_from_dataframe(
    train_df,
    x_col="file_name",
    y_col="label",
    target_size=DatasetConfig.IMAGE_SIZE,
    batch_size=DatasetConfig.BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_dataframe(
    train_df,
    x_col="file_name",
    y_col="label",
    target_size=DatasetConfig.IMAGE_SIZE,
    batch_size=DatasetConfig.BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

if train_generator.n == 0 or val_generator.n == 0:
    raise ValueError("Error: Training or validation generator is empty. Check dataset paths and labels.")

# Build Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=DatasetConfig.EPOCHS
)

# Save Model
model.save(DatasetConfig.MODEL_FILE)

# Plot Training History
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')

    plt.show()

plot_training_history(history)


# Test Predictions
test_df['id'] = test_df['id'].apply(lambda x: os.path.join(DatasetConfig.DATA_PATH, x))
test_df = test_df[test_df['id'].apply(os.path.exists)]

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='id',
    target_size=DatasetConfig.IMAGE_SIZE,
    batch_size=DatasetConfig.BATCH_SIZE,
    class_mode=None,
    shuffle=False
)

if test_generator.n == 0:
    raise ValueError('Error: Test generator is empty.')

test_predictions = model.predict(test_generator)
test_df['prediction'] = (test_predictions > 0.5).astype(int)
test_df.to_csv('test_predictions.csv', index=False)
print('Predictions saved to test_predictions.csv')

# Single Image Prediction
def predict_image(image_path, model_path=DatasetConfig.MODEL_FILE):
    model = tf.keras.models.load_model(model_path)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=DatasetConfig.IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return 'AI-Generated' if prediction > 0.5 else 'Human-Generated'

if not test_df.empty:
    sample_image = test_df.iloc[2]['id']
    print(f'Prediction for sample image: {predict_image(sample_image)}')
