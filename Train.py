
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

IMG_SIZE=(224,224)
BATCH_SIZE=32
EPOCHS=30

datagen=ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.2,rotation_range=30,zoom_range=0.25,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,horizontal_flip=True,brightness_range=[0.7,1.3],fill_mode="nearest")
train_generator=datagen.flow_from_directory("dataset",target_size=IMG_SIZE,batch_size=BATCH_SIZE,class_mode="categorical",subset="training",shuffle=True)
val_generator=datagen.flow_from_directory("dataset",target_size=IMG_SIZE,batch_size=BATCH_SIZE,class_mode="categorical",subset="validation",shuffle=False)
class_weights=dict(enumerate(compute_class_weight(class_weight="balanced",classes=np.unique(train_generator.classes),y=train_generator.classes)))
base_model=MobileNetV2(weights="imagenet",include_top=False,input_shape=(224,224,3)); base_model.trainable=False
x=GlobalAveragePooling2D()(base_model.output); x=Dropout(0.5)(x); x=Dense(256,activation="relu")(x); x=Dropout(0.4)(x)
model=Model(base_model.input,Dense(train_generator.num_classes,activation="softmax")(x))
loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=loss,metrics=["accuracy"])
callbacks=[EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True),ReduceLROnPlateau(monitor="val_loss",factor=0.2,patience=2,min_lr=1e-6),ModelCheckpoint("tea_hibiscus_model.h5",monitor="val_accuracy",save_best_only=True)]
model.fit(train_generator,validation_data=val_generator,epochs=EPOCHS,callbacks=callbacks,class_weight=class_weights)
base_model.trainable=True
for layer in base_model.layers[:-50]: layer.trainable=False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss=loss,metrics=["accuracy"])
model.fit(train_generator,validation_data=val_generator,epochs=10,callbacks=callbacks,class_weight=class_weights)
model.save("tea_hibiscus_model.h5")
