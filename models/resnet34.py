"""

@author: zhiweihu
@create time: 2018-11-26 15:17

"""
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import add, Flatten
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import *
from keras.preprocessing.image import img_to_array, load_img

IMAGE_PATH = "/home/science/huzhiwei/disease/real_datasets"
MODEL_WEIGHTS_PATH = "/home/science/huzhiwei/disease/model/16_resnet34.hdf5"

LEARNING_RATE = 1e-3
BATCH = 16
EPOCH = 30
num_classes = 10


def get_train_test_val():
    """
    获取训练集、验证集以及测试集数据
    :return:
    """
    train_data = np.load(IMAGE_PATH + "/train.npy")
    val_data = np.load(IMAGE_PATH + "/val.npy")
    test_data = np.load(IMAGE_PATH + "/test.npy")
    return train_data, val_data, test_data


def standartize(array):
    array = np.array(array, dtype=np.float32)
    mean = np.mean(array)
    std = np.std(array)
    array -= mean
    array /= std
    return array


def generate_train_batch_data(batch_size, data):
    """
    创建一个batch大小训练集数据
    :param batch_size:
    :param data:
    :return:
    """
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i].split('$')
            image_path = url[0]
            label_index = url[1]
            index = int(label_index) - 1
            batch += 1
            img = load_img(image_path)
            img = img_to_array(img)
            img = standartize(img)
            train_data.append(img)
            class_vector = np.zeros(num_classes, dtype=np.int32)
            class_vector[index] = 1
            train_label.append(class_vector)
            if batch % batch_size == 0:
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0


def generate_val_batch_data(batch_size, data):
    """
    创建一个batch大小验证集数据
    :param batch_size:
    :param data:
    :return:
    """
    while True:
        val_data = []
        val_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i].split('$')
            image_path = url[0]
            label_index = url[1]
            index = int(label_index) - 1
            batch += 1
            img = load_img(image_path)
            img = img_to_array(img)
            img = standartize(img)
            val_data.append(img)
            class_vector = np.zeros(num_classes, dtype=np.int32)
            class_vector[index] = 1
            val_label.append(class_vector)
            if batch % batch_size == 0:
                val_data = np.array(val_data)
                val_label = np.array(val_label)
                yield (val_data, val_label)
                val_data = []
                val_label = []
                batch = 0


def get_test_data():
    """
    获取测试数据
    :return:
    """
    train_data, val_data, test_data = get_train_test_val()
    data = []
    test_label = []
    for i in (range(len(test_data))):
        url = test_data[i].split('$')
        image_path = url[0]
        label_index = url[1]
        index = int(label_index) - 1
        img = load_img(image_path)
        img = img_to_array(img)
        img = standartize(img)
        data.append(img)
        class_vector = np.zeros(num_classes, dtype=np.int32)
        class_vector[index] = 1
        test_label.append(class_vector)
    result_test_data = np.array(data)
    result_test_label = np.array(test_label)
    return result_test_data, result_test_label

def get_one_label_test_data(label_flag):
    """
    获取测试数据
    :return:
    """
    train_data, val_data, test_data = get_train_test_val()
    data = []
    test_label = []
    count = 0
    for i in (range(len(test_data))):
        url = test_data[i].split('$')
        image_path = url[0]
        label_index = url[1]
        index = int(label_index) - 1
        if index != label_flag:
            continue
        count = count + 1
        img = load_img(image_path)
        img = img_to_array(img)
        img = standartize(img)
        data.append(img)
        class_vector = np.zeros(num_classes, dtype=np.int32)
        class_vector[index] = 1
        test_label.append(class_vector)
    result_test_data = np.array(data)
    result_test_label = np.array(test_label)
    return result_test_data, result_test_label


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def resnet34():
    input = Input(shape=(256, 256, 3))
    x = ZeroPadding2D((3, 3))(input)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)
    # model.summary()
    return model

def train():
    """
    训练模型
    :return:
    """
    with tf.device('/cpu:0'):
        model_checkpoint = ModelCheckpoint(str(BATCH) + '_resnet34.hdf5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
        # 自动变化学习率
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1, patience=5, min_lr=0.0001)
        callable = [model_checkpoint, early_stopping, reduce_lr]
        train_data, val_data, test_data = get_train_test_val()
        train_numb = len(train_data)
        valid_numb = len(val_data)
    model = resnet34()

    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)

    with tf.device('/gpu:0'):
        model.fit_generator(generator=generate_train_batch_data(BATCH, train_data), steps_per_epoch=train_numb // BATCH, epochs=EPOCH,
                            verbose=1, validation_data=generate_val_batch_data(BATCH, val_data), validation_steps=valid_numb // BATCH,
                            callbacks=callable, max_q_size=1)


def evaluate():
    test_data, test_label = get_test_data()
    model = resnet34()
    model.load_weights(MODEL_WEIGHTS_PATH)
    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    metrics = model.evaluate(test_data, test_label, batch_size=16)
    print(metrics)


def evaluate_single_class():
    """
    单个类进行预测
    :return:
    """
    for i in range(10):
        print(i+7)
        test_data, test_label = get_one_label_test_data(i)
        model = resnet34()
        model.load_weights(MODEL_WEIGHTS_PATH)
        model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
        metrics = model.evaluate(test_data, test_label, batch_size=16)
        print(metrics)


def predict():
    """
    获取所有测试集上的label以及其预测结果predict
    :return:
    """
    test_data, test_label = get_test_data()
    model = resnet34()
    model.load_weights(MODEL_WEIGHTS_PATH)
    preds = model.predict(test_data, batch_size=BATCH)
    np.save("16_resnet34_label.npy", test_label)
    np.save("16_resnet34_predict.npy", preds)

    return test_label, preds


if __name__ == "__main__":
    # train()
    # evaluate()
    # evaluate_single_class()
    predict()