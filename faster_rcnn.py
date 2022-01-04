import os
import sys
import json
import numpy as np
import time
import pandas as pd

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def nn_base():
    input_shape = (None, None, 3)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    return x

def rpn(base_layers, num_anchors=9):

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]

class RoiPoolingConv(Layer):
    """ROI pooling layer for 2D inputs.
    See Spatial Pyramid pooling in Deep Convolutional Networks for Visual
    Recognition, K. He, X. Zhang, S. Ren, J. Sun
    
    # Arguments
    pool_size: int
            size of pooling region to use, pool_size = 7 will result in a 7x7 region.
    num_rois: number of regions of interest to be used.
    
    # Input shape
        list of two 4D tensors [X_img, X_roi] with shape:
        
     X_img:
         `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
    X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
        
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    """
    
    def __init__(self, pool_size, num_rois, **kwargs):
        
        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        
        self.pool_size = pool_size
        self.num_rois = num_rois
        
        super(RoiPoolingConv, self).__init__(**kwargs)
        
        
    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
            
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]
            
    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels
        
     
    def call(self, x, mask=None):
        assert(len(x) == 2)
        
        img = x[0]
        rois = x[1]
        
        input_shape = K.shape(img)
        
        outputs = []
        
        for roi_idx in range(self.num_rois):
            
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]
            
            row_length = w / float(self.pool_size)
            col_length = h / float(self.pool_size)
            
            num_pool_regions = self.pool_size
            
            #NOTE: the RoiPooling implementation differs between theano and tensorflow due to the lack of a resize op
            # in theano. The theano implementation is much less efficient and leads to long compile times

            if self.dim_ordering == 'th':
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = x + ix * row_length
                        x2 = x1 + row_length
                        y1 = y + jy * col_length
                        y2 = y1 + col_length
                        
                        x1 = K.cast(x1, 'int32')
                        x2 = K.cast(x2, 'int32')
                        y1 = K.cast(y1, 'int32')
                        y2 = K.cast(y2, 'int32')
                        
                        x2 = x1 + K.maximum(1, x2-x1)
                        y2 = y1 + K.maximum(1, y2-y1)
                        
                        new_shape = [input_shape[0], input_shape[1],
                                    y2 - y1, x2 - x1]
                        
                        x_crop = img[:, :, y1:y2, x1:x2]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(2, 3))
                        outputs.append(pooled_val)
                        
            elif self.dim_ordering == 'tf':
                x = K.cast(x, 'int32')
                y = K.cast(y, 'int32')
                w = K.cast(w, 'int32')
                h = K.cast(h, 'int32')
                
                rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
                outputs.append(rs)
                
        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        
        if self.dim_ordering == 'th':
            final_output = K.permute_dimensions(final_output, (0, 1, 4, 2, 3))
            
        else:
            final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))
            
        return final_output
        
    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def classifier(base_layers, input_rois, num_rois, nb_classes = 21, trainable=False):

    pooling_regions = 7
    input_shape = (num_rois,7,7,512)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]

def build_faster_rcnn():
    x = nn_base()
    rpn_output = rpn(x)
    classifier_output = classifier(x, )
    

    return model

def main():
    scaphoid_images_dir = 'scaphoid_pictures'
    train_dataset_csv = 'cv1_train.csv'
    val_dataset_csv = 'cv1_test.csv'
    train_dataset = pd.read_csv(train_dataset_csv)
    val_dataset = pd.read_csv(val_dataset_csv)

    train_aug_gen =  ImageDataGenerator().flow_from_dataframe(train_dataset, scaphoid_images_dir, 
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

    val_aug_gen = ImageDataGenerator().flow_from_dataframe(val_dataset, scaphoid_images_dir, 
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

if __name__ == '__main__':
    main()