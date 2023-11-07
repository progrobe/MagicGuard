from modulefinder import Module
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from frontier_stitching import gen_adversaries, verify,test_acc
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_utils import *
import argparse
from keras.layers import Lambda
from keras.models import Input, Model
from tensorflow.python.keras import backend

os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def sin_layer(x):
    x = x + 0.01*backend.sin(10000*x)
    # x = x*x
    return x

def to_float(x, y):
    return tf.cast(x, tf.float32) / 255.0, y

def comp(model):
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=["accuracy"])

def embed(model):
    dataset = tf.keras.preprocessing.image_dataset_from_directory('/home/mist/relight/ziyi/data/train', 
                                                                    label_mode = 'categorical',
                                                                    image_size = (224, 224),
                                                                    subset = 'training',
                                                                    batch_size = 32,
                                                                    seed = 123,
                                                                    validation_split = 0.2
                                                                )
    val_set = tf.keras.preprocessing.image_dataset_from_directory('/home/mist/relight/ziyi/data/train', 
                                                                    label_mode = 'categorical',
                                                                    image_size = (224, 224),
                                                                    subset = 'validation',
                                                                    batch_size = 32,
                                                                    seed = 123,
                                                                    validation_split = 0.2
                                                                )
                                                            
    test_set = tf.keras.preprocessing.image_dataset_from_directory('/home/mist/relight/ziyi/data/test', 
                                                                    label_mode = 'categorical',
                                                                    image_size = (224, 224),
                                                                    batch_size = 32,
                                                                    seed = 123,
                                                                )

    dataset = dataset.map(to_float)
    val_set = val_set.map(to_float)
    test_set = test_set.map(to_float)

    l = 30

    # generate key set
    true_advs, false_advs = gen_adversaries(model, l, dataset, 0.01)

    # In case that not the full number of adversaries could be generated a reduced amount is returned
    assert (len(true_advs + false_advs) == l)

    # 拼接trueadv和falseadv
    key_set_x = tf.data.Dataset.from_tensor_slices(
        [x for x, y in true_advs + false_advs])
    key_set_y = tf.data.Dataset.from_tensor_slices(
        [y for x, y in true_advs + false_advs])
    key_set = tf.data.Dataset.zip((key_set_x, key_set_y)).batch(32)

    _, acc = model.evaluate(key_set, batch_size=32)
    print("wm :", acc)
    _, acc = model.evaluate(test_set, batch_size=32)
    print("test:", acc)

    model.fit(key_set, epochs=14, validation_data=val_set)

    _, acc = model.evaluate(key_set, batch_size=32)
    print("wm :", acc)
    _, acc = model.evaluate(test_set, batch_size=32)
    print("test:", acc)

def finetune(model):
    dataset = tf.keras.preprocessing.image_dataset_from_directory('/home/mist/relight/ziyi/data/train', 
                                                                    label_mode = 'categorical',
                                                                    image_size = (224, 224),
                                                                    subset = 'training',
                                                                    batch_size = 32,
                                                                    seed = 123,
                                                                    validation_split = 0.2
                                                                )
    val_set = tf.keras.preprocessing.image_dataset_from_directory('/home/mist/relight/ziyi/data/train', 
                                                                    label_mode = 'categorical',
                                                                    image_size = (224, 224),
                                                                    subset = 'validation',
                                                                    batch_size = 32,
                                                                    seed = 123,
                                                                    validation_split = 0.2
                                                                )
                                                            
    test_set = tf.keras.preprocessing.image_dataset_from_directory('/home/mist/relight/ziyi/data/test', 
                                                                    label_mode = 'categorical',
                                                                    image_size = (224, 224),
                                                                    batch_size = 32,
                                                                    seed = 123,
                                                                )

    dataset = dataset.map(to_float)
    val_set = val_set.map(to_float)
    test_set = test_set.map(to_float)

    og_model = keras.models.load_model('/home/mist/relight/ziyi/model/vgg16_wm_text')

    wmtest2(og_model, model)
    _, acc = model.evaluate(test_set, batch_size=32)
    print("test:", acc)
    
    train_x = np.array([x for bx, by in dataset for x in bx])

    train_y = np.array([y for bx, by in dataset for y in by])
    print(train_x.shape)
    # 数据增强可以干扰adv wm
    datagen = image.ImageDataGenerator(horizontal_flip=True, 
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    rotation_range=15,
                                    )
                    # True 0.2 0.2 30
    datagen.fit(train_x)
    train_datagen = datagen.flow(train_x, train_y,
                                 batch_size=32,
                                )

    for i in range(10):
        model.fit(train_datagen, epochs=1, validation_data=val_set)
        wmtest2(og_model, model)
        _, acc = model.evaluate(test_set, batch_size=32)
        print("test:", acc)
        model.fit(val_set, epochs=1, validation_data=val_set)
        wmtest2(og_model, model)
        _, acc = model.evaluate(test_set, batch_size=32)
        print("test:", acc)
        
        

def wmtest2(og_model, model):
    dataset = tf.keras.preprocessing.image_dataset_from_directory('/home/mist/relight/ziyi/data/train', 
                                                                    label_mode = 'categorical',
                                                                    image_size = (224, 224),
                                                                    subset = 'training',
                                                                    batch_size = 32,
                                                                    seed = 123,
                                                                    validation_split = 0.2
                                                                )
    val_set = tf.keras.preprocessing.image_dataset_from_directory('/home/mist/relight/ziyi/data/train', 
                                                                    label_mode = 'categorical',
                                                                    image_size = (224, 224),
                                                                    subset = 'validation',
                                                                    batch_size = 32,
                                                                    seed = 123,
                                                                    validation_split = 0.2
                                                                )
    test_set = tf.keras.preprocessing.image_dataset_from_directory('/home/mist/relight/ziyi/data/test', 
                                                                    label_mode = 'categorical',
                                                                    image_size = (224, 224),
                                                                    batch_size = 32,
                                                                    seed = 123,
                                                                )

    dataset = dataset.map(to_float)
    val_set = val_set.map(to_float)
    test_set = test_set.map(to_float)

    l = 30

    # generate key set
    true_advs, false_advs = gen_adversaries(og_model, l, dataset, 0.01)

    # In case that not the full number of adversaries could be generated a reduced amount is returned
    assert (len(true_advs + false_advs) == l)

    # 拼接trueadv和falseadv
    key_set_x = tf.data.Dataset.from_tensor_slices(
        [x for x, y in true_advs + false_advs])
    key_set_y = tf.data.Dataset.from_tensor_slices(
        [y for x, y in true_advs + false_advs])
    key_set = tf.data.Dataset.zip((key_set_x, key_set_y)).batch(32)

    _, acc = model.evaluate(key_set, batch_size=32)
    print("wm :", acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='', help='architecture of the the model')
    parser.add_argument('--test_ds', default='/home/mist/relight/ziyi/watermark/adversarial/adversarial-frontier-stitching/adv/', help='the path of the test data')
    args = parser.parse_args()

    # model = keras.models.load_model('/home/mist/relight/ziyi/model/vgg16_wm_text')
    # embed(model)
    # model.save('/home/mist/magic_guard2/watermark/adversarial/adversarial-frontier-stitching/checkpoint/vgg16')
    # print('----------------- finished embedding----------------------')

    model = keras.models.load_model('/home/mist/magic_guard2/watermark/adversarial/adversarial-frontier-stitching/checkpoint/vgg16')

    # x = model.output
    # x = Lambda(sin_layer, name='sin_layer')(x)

    # model_sin = keras.Model(inputs=model.input, outputs=x)
    # model_sin.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # finetune(model_sin)

    # og_model = keras.models.load_model('/home/mist/relight/ziyi/model/vgg16_wm_text')
    # wmtest2(og_model,model_sin)

    finetune(model)

    og_model = keras.models.load_model('/home/mist/relight/ziyi/model/vgg16_wm_text')
    wmtest2(og_model,model)

