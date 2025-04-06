import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

# Paths
train_dir = '/Users/rishabhbhargav/PycharmProjects/HackUSF/model_training/data/train/'
test_dir = '/Users/rishabhbhargav/PycharmProjects/HackUSF/model_training/data/test/'

# Load datasets
batch_size = 32
img_size = (224, 224)

train_ds = image_dataset_from_directory(
    train_dir,
    label_mode='binary',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True
)

val_ds = image_dataset_from_directory(
    test_dir,
    label_mode='binary',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=False
)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Check the distribution of labels in the training set
y_train = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
y_train = y_train.flatten()  # Convert to 1D array

unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique, counts)
plt.title('Class Distribution in Training Data')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weight_dict)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomHeight(0.2),  # Added new augmentations
    tf.keras.layers.RandomWidth(0.2),
])

def preprocess_with_aug(image, label):
    return data_augmentation(image, training=True), label

augmented_train_ds = train_ds.map(preprocess_with_aug, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)

# Base Model: EfficientNetB1
base_model = tf.keras.applications.EfficientNetB1(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
base_model.trainable = True

# Freeze most layers to prevent overfitting
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Classification Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Focal Loss (adjusted for better balance)
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.8, name='focal_loss'):  # Increased gamma, adjusted alpha
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = self.alpha * y_true * tf.pow(1 - y_pred, self.gamma) + \
                 (1 - self.alpha) * (1 - y_true) * tf.pow(y_pred, self.gamma)
        return tf.reduce_mean(weight * ce)

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=FocalLoss(gamma=2.0, alpha=0.8),  # Adjusted loss function
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint('balanced_benign_malignant.keras', save_best_only=True)

# Train Model
history = model.fit(
    augmented_train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weight_dict
)

# Save Final Model
model.save('final_balanced_model.keras')

# To load later:
# model = tf.keras.models.load_model('final_balanced_model.keras', custom_objects={'FocalLoss': FocalLoss})

# After training, evaluate the model performance using classification report
y_pred = model.predict(val_ds)
y_pred = (y_pred > 0.5)  # Convert to binary predictions

from sklearn.metrics import classification_report

y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
print(classification_report(y_true, y_pred))