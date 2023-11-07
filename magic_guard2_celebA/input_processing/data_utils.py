import numpy as np
import os
import pathlib
from keras_vggface import utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


img_height = 224
img_width = 224


def load_dataset(args, test=False):
    data_gen = ImageDataGenerator(
                        rescale=1./255,
                        validation_split=0.2,
                        preprocessing_function=lambda x: utils.preprocess_input(x, version= 1 if args.model == 'vgg16' else 2))
    if test == False:
        train_dir = pathlib.Path(args.train_ds)
        train_ds = data_gen.flow_from_directory(
            train_dir,
            class_mode='categorical',
            target_size=(img_height, img_width),
            batch_size = args.batch_size,
            subset='training',
        )

        valid_ds = data_gen.flow_from_directory(
            train_dir,
            class_mode='categorical',
            target_size=(img_height, img_width),
            batch_size = args.batch_size,
            subset='validation',
        )
        
        return train_ds, valid_ds
    else:
        test_dir = pathlib.Path(args.test_ds)
        #args.test_ds
        test_ds = data_gen.flow_from_directory(
            test_dir,
            class_mode='categorical',
            target_size=(img_height, img_width),
            batch_size = args.batch_size
        )
        return test_ds
