import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input
import matplotlib.pyplot as plt
from convert import convert_model


flowers_path = './flowers/flowers'
classes = os.listdir(flowers_path)

image_size = (128, 128)
input_shape = (128,128, 3)
batch_size = 32

# Image generator configuration, it is a pre-processing
# Later it can be used with the "take" method
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    flowers_path,
    validation_split=0.2,          
    subset="training",             
    seed=42,                     
    image_size=image_size,         
    batch_size=batch_size,         
    label_mode="categorical",      
    class_names=classes
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    flowers_path,
    validation_split=0.2,
    subset="validation",         
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
    class_names=classes
)


# Define your preprocessing function
def preprocess_image(image, label):
    image = preprocess_input(image)
    return image, label


def get_model():
    # Get base model 
    base_model = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape)
    # Freeze the layers in base model
    for layer in base_model.layers:
        layer.trainable = False
    # Add new layers
    x = Flatten()(base_model.output)
    x = Dense(500, activation='relu', name='fc1')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='softmax', name='fc2')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model


def data_augmentation(x):
    augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2)     
    ]
    for layer in augmentation_layers:
        x = layer(x)
    return x


def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()


train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

#apply augmentation
train_ds_aug = train_ds.map(lambda x, y: (data_augmentation(x), y))
model = get_model()
opt = Adam(lr=1e-3, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
chechpoint = keras.callbacks.ModelCheckpoint(
    'xception_v2_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
history_fine_aug=model.fit(train_ds_aug, epochs=20, verbose=1, validation_data=val_ds, callbacks=[chechpoint,early_stopping])


show_train_history(history_fine_aug,'loss','val_loss')
show_train_history(history_fine_aug,'accuracy','val_accuracy')

# Save the trained model
KERAS_MODEL_PATH='./models/xception_v2_07_0.813.h5'
TFLITE_MODEL_PATH='./models/flower-model.tflite'
print('... saving model `to ./models')
model.save(KERAS_MODEL_PATH)

# Convert model to TF-lite
convert_model(KERAS_MODEL_PATH, TFLITE_MODEL_PATH)