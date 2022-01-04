import os
import sys
import json
import numpy as np
import time
import pandas as pd

scaphoid_images_dir = 'scaphoid_pictures'
train_dataset_csv = 'cv1_train.csv'
val_dataset_csv = 'cv1_test.csv'
train_dataset = pd.read_csv(train_dataset_csv)
val_dataset = pd.read_csv(val_dataset_csv)

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import disable_eager_execution
from tensorflow.keras.applications import EfficientNetB3

#disable_eager_execution()

'''
train_datagen = ImageDataGenerator(
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=True
                    )
train_aug_gen = train_datagen.flow_from_dataframe(train_dataset, scaphoid_images_dir, 
                                  x_col='Filename', y_col=['bbox_topleft_x', 'bbox_topleft_y', 'bbox_botright_x', 'bbox_botright_y'], 
                                  has_ext=True, 
                                  target_size=(960, 1280), 
                                  color_mode='rgb', 
                                  classes=None, 
                                  class_mode="other", 
                                  batch_size=4, 
                                  shuffle=True, 
                                  seed=None, 
                                  interpolation='nearest',
                                  validate_filenames=False
                                  
                                 )
'''                                 
train_aug_gen =  ImageDataGenerator().flow_from_dataframe(train_dataset, scaphoid_images_dir, 
                                  x_col='Filename', y_col=['bbox_topleft_x', 'bbox_topleft_y', 'bbox_botright_x', 'bbox_botright_y'], 
                                  has_ext=True, 
                                  target_size=(480, 640), 
                                  color_mode='rgb', 
                                  classes=None, 
                                  class_mode="other", 
                                  batch_size=4, 
                                  shuffle=True, 
                                  seed=None, 
                                  interpolation='nearest',
                                  validate_filenames=False
                                  
                                 )   

'''
val_datagen = ImageDataGenerator(
                    featurewise_center=False,
                    featurewise_std_normalization=False,
                    rotation_range=0,
                    width_shift_range=0,
                    height_shift_range=0,
                    horizontal_flip=False
                    )
val_aug_gen = train_datagen.flow_from_dataframe(val_dataset, scaphoid_images_dir, 
                                  x_col='Filename', y_col=['bbox_topleft_x', 'bbox_topleft_y', 'bbox_botright_x', 'bbox_botright_y'], 
                                  has_ext=True, 
                                  target_size=(960, 1280), 
                                  color_mode='rgb', 
                                  classes=None, 
                                  class_mode="other", 
                                  batch_size=4, 
                                  shuffle=False, 
                                  seed=None, 
                                  interpolation='nearest',
                                  validate_filenames=False
                                  
                                 )
'''
val_aug_gen = ImageDataGenerator().flow_from_dataframe(val_dataset, scaphoid_images_dir, 
                                  x_col='Filename', y_col=['bbox_topleft_x', 'bbox_topleft_y', 'bbox_botright_x', 'bbox_botright_y'], 
                                  has_ext=True, 
                                  target_size=(480, 640), 
                                  color_mode='rgb', 
                                  classes=None, 
                                  class_mode="other", 
                                  batch_size=4, 
                                  shuffle=False, 
                                  seed=None, 
                                  interpolation='nearest',
                                  validate_filenames=False
                                  
                                 )
# model initialization
# load the VGG16 network, ensuring the head FC layers are left off
resnet = EfficientNetB3(weights="imagenet", include_top=False, input_tensor=Input(shape=(480, 640, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training processr
resnet.trainable = False
# flatten the max-pooling output of VGG
flatten = resnet.output
flatten = Flatten()(flatten)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)
# construct the model we will fine-tune for bounding box regression
model = Model(inputs=resnet.input, outputs=bboxHead)
opt = Adam(lr=1e-5)
model.compile(loss="mae", optimizer=opt)

callbacks = [
    ModelCheckpoint("fold1_scaphoid_detection.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
batch_size = 16
n_epochs = 100
steps_per_epoch = 160// train_aug_gen.batch_size
validation_steps = 80 // val_aug_gen.batch_size
'''
for epoch_num in range(n_epochs):
    X, Y = next(train_aug_gen)
    loss = model.train_on_batch(X, Y)
    print(loss)
'''
history = model.fit(x = train_aug_gen,
                validation_data = val_aug_gen,
                steps_per_epoch = steps_per_epoch,
                validation_steps = validation_steps,
                epochs = n_epochs,
                verbose = True,
                callbacks=callbacks)