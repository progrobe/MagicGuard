from __future__ import (absolute_import, division, print_function)
from tensorflow import keras

from tensorflow.keras import Model, regularizers, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend

from keras_vggface.vggface import VGGFace
from keras_vggface import utils

import argparse
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from data_utils import *


def main(args):
    train_ds, valid_ds = load_dataset(args)
    nb_class = 30

    if args.model == 'vgg16':
        vgg_model = VGGFace(include_top=False, model='vgg16', input_shape=(224, 224, 3),weights='vggface')

        for layer in vgg_model.layers:
            layer.trainable = True

        last_layer = vgg_model.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        # x = Dense(args.hidden_size, activation='relu', name='fc6')(x)
        # x = Dense(args.hidden_size, activation='relu', name='fc7')(x)
        out = Dense(nb_class, activation='softmax', name='fc8')(x)
        custom_vgg_model = Model(vgg_model.input, out)

    elif args.model == 'resnet50' or args.model == 'senet50':
        vgg_model = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
        # for layer in vgg_model.layers:
        #     layer.trainable = False 
        
        last_layer = vgg_model.get_layer('avg_pool').output
        x = Flatten(name='flatten')(last_layer)
        out = Dense(nb_class, activation='softmax', name='classifier')(x)
        custom_vgg_model = Model(vgg_model.input, out)


    model = Sequential()
    model.add(custom_vgg_model)
    # import ssl
    # ssl._create_default_https_context = ssl._create_unverified_context
    # model = keras.applications.VGG16(input_shape = (224,224,3), weights='imagenet', classes=nb_class, include_top=False)

    # model = keras.applications.DenseNet121(input_shape = (224,224,3), weights=None, classes=nb_class)
    # model = keras.models.load_model('/wangrun/ziyi/model/resnet50_wm_text')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint(
                                filepath='model/vgg16_best/',
                                monitor='val_accuracy',
                                mode='max',
                                save_best_only=True)

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=args.lr),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.summary()

    # train the model on new data
    initial_epochs = 30

    _ = model.fit(train_ds,
                steps_per_epoch=train_ds.samples / train_ds.batch_size,
                epochs=initial_epochs, 
                callbacks=[early_stopping, model_checkpoint],
                validation_data=valid_ds,
                validation_steps=valid_ds.samples / valid_ds.batch_size,
                verbose=1)

    model.save(os.path.join(args.save_dir, args.save_model))

def test(args):
    # test_ds = load_dataset(args, test=True)
    test_ds = load_dataset(args, test=True)
    #model = keras.models.load_model('/wangrun/ziyi/model/vgg16_best')
    model = keras.models.load_model('/wangrun/ziyi/model/resnet50_wm_text')
    # resnet50 测试集准确度92%
    res = model.evaluate(test_ds, batch_size=16)
    # batch_size = 16
    print("test loss, test acc:", res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ds', default='/wangrun/ziyi/data/train', help='the path of the training data')
    parser.add_argument('--test_ds', default='/wangrun/ziyi/data/test', help='the path of the test data')
    parser.add_argument('--wm_ds', default='/wangrun/ziyi/data/wm_content_relight_left', help='the path of the watermarked set')
    parser.add_argument('--wm_lbl', help='the path of the watermark random label file')

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--hidden_size', default=512, type=int, help='hidden size')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    
    parser.add_argument('--save_dir', default='/wangrun/ziyi/model/', help='the path to the model dir')
    parser.add_argument('--save_model', default='vgg16', help='model name')

    parser.add_argument('--load_path', default='', help='the path to the pre-trained model, to be used with resume flag')
    parser.add_argument('--model', default='vgg16', help='architecture of the the model')

    args = parser.parse_args()
    main(args)
    # test(args)
