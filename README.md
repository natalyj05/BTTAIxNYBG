# BTTAIxNYBG
## Context
Business context: https://www.kaggle.com/competitions/bttai-nybg-2024 <br/>
Data context: https://www.kaggle.com/competitions/bttai-nybg-2024/data

## Overview of the Approach


## Details of the Submission
### Importing Libraries
Libraries necessary for handling files, data manipulation, image processing, and deep learning are imported.

```
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```

### Defining Directories
Paths to the training, testing, and validation data directories are defined.

```
train_dir = '/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-train/BTTAIxNYBG-train'
test_dir = '/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-test/BTTAIxNYBG-test'
validation_dir = '/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-validation/BTTAIxNYBG-validation'
```

### Loading Dataframes
CSV files that presumably contain metadata about the images, such as filenames (imageFile) and class labels (classLabel), are loaded into pandas data frames.

```
train_df = pd.read_csv('/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-train.csv')
test_df = pd.read_csv('/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-test.csv')
validate_df = pd.read_csv('/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-validation.csv')
```

### Data Augmentation Configuration
For the training set, various transformations like rotation, shifting, shearing, zooming, flipping, and changes in brightness and channel intensity are set up. 

```
train_datagen = ImageDataGenerator(
    rotation_range=40,  # Degrees of random rotations
    width_shift_range=0.2,  # Fraction of total width, for horizontal shift
    height_shift_range=0.2,  # Fraction of total height, for vertical shift
    shear_range=0.2,  # Shear Intensity (Shear angle in counter-clockwise direction)
    zoom_range=[0.8, 1.2],  # Range for random zoom. Now allows for zoom in and out
    horizontal_flip=True,  # Randomly flip inputs horizontally
    fill_mode='nearest',  # Strategy to fill in newly created pixels
    brightness_range=[0.5, 1.5],  # Randomly alter the brightness of images
    channel_shift_range=50.0,  # Range for random channel shifts
    rescale=1./255  # Rescaling factor for normalizing pixel values
)
```

This helps prevent overfitting by artificially expanding the dataset with variations of training images. Validation and test data generators only perform value rescaling to match the model’s expected input format.

```
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
```

### Data Generators
The function df_to_dataset is defined to convert data frames into data generators. These generators produce batches of images, which are fed into the model during training or evaluation.

```
def df_to_dataset(dataframe, datagen, directory, batch_size=32):
    return datagen.flow_from_dataframe(
        dataframe=dataframe,
        directory=directory,
        x_col='imageFile',  # Column in dataframe that contains the filenames
        y_col='classLabel',  # Column in dataframe that contains the class/label
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'  # Multiclass classification
    )
```

Data generators for training and validation datasets are created using this function.

```
train_dataset = df_to_dataset(train_df, train_datagen, train_dir)
validation_dataset = df_to_dataset(validate_df, validation_datagen, validation_dir)
```

### Model Setup
The MobileNetV2 architecture, pre-trained on the ImageNet dataset, is loaded and modified. Layers up to a certain point (fine_tune_at) are frozen to keep their learned weights.

```
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
```

New layers are added on top of MobileNetV2 for the specific classification task: a global average pooling layer to reduce feature dimensions, a dense layer for feature interpretation, and a final softmax layer for classification output across 10 classes.

```
model = Sequential([
    base_model,
    # Convert features to vectors
    tf.keras.layers.GlobalAveragePooling2D(),
    # Add a dense layer for classification
    Dense(1024, activation='relu'),
    # Final layer with softmax activation for multi-class classification
    Dense(10, activation='softmax')
])
```


### Model Compilation
An exponential decay learning rate schedule is defined for the optimizer.

```
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9
)
```

The model is compiled with the Adam optimizer, categorical cross-entropy loss function, and accuracy as the performance metric.

```
model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])
```

### Model Training
Initially, the model is trained on a subset of the training data for quick testing with fewer epochs to debug the setup.

```
train_subset = train_df.sample(frac=0.7, random_state=42)
train_subset_dataset = df_to_dataset(train_subset, train_datagen, train_dir)
```

Callbacks for early stopping (to prevent overfitting) and model checkpointing (to save the best-performing model) are defined and used.

```
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
]

history = model.fit(
    train_subset_dataset,  # Use the subset of data
    validation_data=validation_dataset,
    epochs=3,  # Initially train for fewer epochs for debugging
    callbacks=callbacks
)
```

After initial testing, the model is trained on the entire training dataset.

```
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=4,
    callbacks=callbacks
)
```

### Model Evaluation
After training, the model's performance is evaluated on the validation dataset.

```
validation_loss, validation_accuracy = model.evaluate(validation_dataset)
print(f'Validation Loss: {validation_loss}')
print(f'Validation Accuracy: {validation_accuracy}')
```

### Prediction and Submission
The model makes predictions on the test set, which does not include labels (class_mode=None).

```
test_dataset = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col='imageFile',  # Make sure column name matches test_df column name for filenames
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,  # No labels
    shuffle=False
)
predictions = model.predict(test_dataset)
predicted_class_indices = np.argmax(predictions, axis=1)
```

Predictions are converted to class indices, and a submission file is created mapping test image IDs to predicted class labels, which is then saved as a CSV file for submission.

```
submission_df = pd.DataFrame({'uniqueID': test_df['uniqueID'], 'classID': predicted_class_indices})
submission_df.to_csv('/kaggle/working/submission.csv', index=False)
```
