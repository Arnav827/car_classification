import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
import pathlib
from tensorflow.keras import regularizers

data_dir = pathlib.Path('/cs/snapless/gabis/nfun/arbel/images/Cars_Body_Type')

original_model = tf.keras.applications.ResNet50V2(include_top=False)
original_model.trainable = True
for layer in original_model.layers[:-60]:
    layer.trainable = False

BATCH_SIZE = 64
IMG_SIZE = 224
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  label_mode='categorical',
  image_size=(IMG_SIZE, IMG_SIZE),
  batch_size=BATCH_SIZE)

test_val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  label_mode='categorical',
  image_size=(IMG_SIZE, IMG_SIZE),
  batch_size=BATCH_SIZE)

val_ds=test_val_ds.shard(num_shards=2, index=0)
test_ds=test_val_ds.shard(num_shards=2, index=1)

data_augmentation_model = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

model_inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation_model(model_inputs)
x = tf.keras.applications.resnet_v2.preprocess_input(x)
x = original_model(x)
x  = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = tf.keras.layers.Dropout(0.3)(x)
early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
model_outputs=tf.keras.layers.Dense(7, activation=tf.keras.activations.softmax)(x)

new_model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs)
new_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(),metrics=[tf.keras.metrics.CategoricalAccuracy()])
initial_epochs = 100
history = new_model.fit(train_ds,epochs=initial_epochs,validation_data=val_ds,callbacks=[early_stop, reduce_lr])

new_model.save('./model.keras')