"""

@author: zhiweihu
@create time: 2018-11-12 15:22
@site: resnet结构解析网站：https://blog.csdn.net/zjucor/article/details/78636573?locationNum=2&fps=1
"""
import keras.backend as K
import numpy as np

from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, Activation, AveragePooling2D, Multiply, Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import *
from keras.layers import ZeroPadding2D
from keras import layers
from keras.applications.resnet50 import ResNet50
from keras.layers.merge import concatenate
from keras import regularizers


IMAGE_PATH = "/home/science/huzhiwei/disease/real_datasets"
MODEL_WEIGHTS_PATH = "/home/science/huzhiwei/disease/model/16_attention_resnet.hdf5"


LEARNING_RATE = 1e-3
BATCH = 16
EPOCH = 100
weight_decay = 5e-4
num_classes = 10


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    :param input_tensor:
    :param kernel_size:
    :param filters:
    :param stage:
    :param block:
    :return:
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'my_res' + str(stage) + block + '_branch'
    bn_name_base = 'my_bn' + str(stage) + block + '_branch'
    relu_name_base = 'my_relu' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu', name=relu_name_base + '2a')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu', name=relu_name_base + '2b')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    low_layer = input_tensor
    high_layer = x

    merge_layer = concatenate([low_layer, high_layer], axis=3)
    global_avg_pool = AveragePooling2D(pool_size=(int(merge_layer.shape[1]), int(merge_layer.shape[2])),
                                       strides=(int(merge_layer.shape[1]), int(merge_layer.shape[2])))(merge_layer)
    conv_layer1 = Conv2D(filters3, (1, 1), padding="same", kernel_regularizer=regularizers.l2(weight_decay))(
        global_avg_pool)
    conv_layer1 = BatchNormalization()(conv_layer1)
    conv_layer1 = Activation(activation='relu')(conv_layer1)
    conv_layer2 = Conv2D(int(low_layer.shape[3]), (1, 1), padding="same",
                         kernel_regularizer=regularizers.l2(weight_decay))(conv_layer1)
    conv_layer2 = BatchNormalization()(conv_layer2)
    conv_layer2 = Activation(activation='sigmoid')(conv_layer2)
    multi = Multiply()([low_layer, conv_layer2])

    x = layers.add([x, multi])
    x = Activation('relu', name=relu_name_base + '2c')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    conv_name_base = 'my_res' + str(stage) + block + '_branch'
    bn_name_base = 'my_bn' + str(stage) + block + '_branch'
    relu_name_base = 'my_relu' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu', name=relu_name_base + '2a')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu', name=relu_name_base + '2b')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu', name=relu_name_base + '2c')(x)
    return x


def pre_resNet50():
    """
    构建resnet50模型用于迁移学习其权重参数
    :return:
    """
    img_input = Input((256, 256, 3))

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='my_conv1')(x)
    x = BatchNormalization(name='my_bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    output = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(output)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax', name="predictions")(x)
    model = Model(inputs=img_input, outputs=predictions)

    return model


def get_conv_layer_name_list(stage, block, flag=0):
    name_list = []
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    relu_name_base = 'relu' + str(stage) + block + '_branch'
    if flag == 1:
        conv_name_base = 'my_' + conv_name_base
        bn_name_base = 'my_' + bn_name_base
        relu_name_base = 'my_' + relu_name_base

    name_list.append(conv_name_base + '2a')
    name_list.append(bn_name_base + '2a')
    # name_list.append(relu_name_base + '2a')

    name_list.append(conv_name_base + '2b')
    name_list.append(bn_name_base + '2b')
    # name_list.append(relu_name_base + '2b')

    name_list.append(conv_name_base + '2c')
    name_list.append(bn_name_base + '2c')

    name_list.append(conv_name_base + '1')
    name_list.append(bn_name_base + '1')

    # name_list.append(relu_name_base + '2c')

    return name_list


def get_identity_layer_name_list(stage, block, flag=0):
    name_list = []
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    relu_name_base = 'relu' + str(stage) + block + '_branch'

    if flag == 1:
        conv_name_base = 'my_' + conv_name_base
        bn_name_base = 'my_' + bn_name_base
        relu_name_base = 'my_' + relu_name_base

    name_list.append(conv_name_base + '2a')
    name_list.append(bn_name_base + '2a')
    # name_list.append(relu_name_base + '2a')

    name_list.append(conv_name_base + '2b')
    name_list.append(bn_name_base + '2b')
    # name_list.append(relu_name_base + '2b')

    name_list.append(conv_name_base + '2c')
    name_list.append(bn_name_base + '2c')
    # name_list.append(relu_name_base + '2c')

    return name_list


def get_layer_name_dict(flag=0):
    layer_names = []
    conv1_name = "conv1"
    bn1_name = "bn_conv1"
    if flag == 1:
        conv1_name = "my_" + conv1_name
        bn1_name = "my_" + bn1_name

    layer_names.append(conv1_name)
    layer_names.append(bn1_name)

    name_list = get_conv_layer_name_list(stage=2, block='a', flag=flag)
    layer_names = layer_names + name_list
    name_list = get_identity_layer_name_list(stage=2, block='b', flag=flag)
    layer_names = layer_names + name_list
    name_list = get_identity_layer_name_list(stage=2, block='c', flag=flag)
    layer_names = layer_names + name_list

    name_list = get_conv_layer_name_list(stage=3, block='a', flag=flag)
    layer_names = layer_names + name_list
    name_list = get_identity_layer_name_list(stage=3, block='b', flag=flag)
    layer_names = layer_names + name_list
    name_list = get_identity_layer_name_list(stage=3, block='c', flag=flag)
    layer_names = layer_names + name_list
    name_list = get_identity_layer_name_list(stage=3, block='d', flag=flag)
    layer_names = layer_names + name_list

    name_list = get_conv_layer_name_list(stage=4, block='a', flag=flag)
    layer_names = layer_names + name_list
    name_list = get_identity_layer_name_list(stage=4, block='b', flag=flag)
    layer_names = layer_names + name_list
    name_list = get_identity_layer_name_list(stage=4, block='c', flag=flag)
    layer_names = layer_names + name_list
    name_list = get_identity_layer_name_list(stage=4, block='d', flag=flag)
    layer_names = layer_names + name_list
    name_list = get_identity_layer_name_list(stage=4, block='e', flag=flag)
    layer_names = layer_names + name_list
    name_list = get_identity_layer_name_list(stage=4, block='f', flag=flag)
    layer_names = layer_names + name_list

    name_list = get_conv_layer_name_list(stage=5, block='a', flag=flag)
    layer_names = layer_names + name_list
    name_list = get_identity_layer_name_list(stage=5, block='b', flag=flag)
    layer_names = layer_names + name_list
    name_list = get_identity_layer_name_list(stage=5, block='c', flag=flag)
    layer_names = layer_names + name_list

    return layer_names


def attention_resnet():
    """
    定义自己的resnet模型,去除原始resnet的最后全连接层
    :param model:
    :return:
    """
    input_tensor = Input(shape=(256, 256, 3))
    base_model = ResNet50(input_tensor=input_tensor, include_top=False, weights="imagenet")
    model = pre_resNet50()
    base_layer_list_names = get_layer_name_dict()
    new_layer_list_names = get_layer_name_dict(flag=1)
    base_layer_dict_names = dict()

    for i in range(len(base_layer_list_names)):
        base_layer_dict_names[new_layer_list_names[i]] = base_layer_list_names[i]

    for layer in model.layers:
        if layer.name in base_layer_dict_names:
            layer_name = base_layer_dict_names[layer.name]
            layer.set_weights(base_model.get_layer(layer_name).get_weights())
            layer.trainable = False

    return model


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


def train():
    """
    训练模型
    :return:
    """
    with tf.device('/cpu:0'):
        model_checkpoint = ModelCheckpoint(str(BATCH) + '_attention_resnet.hdf5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
        # 自动变化学习率
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1, patience=5, min_lr=0.0001)
        callable = [model_checkpoint, early_stopping, reduce_lr]
        train_data, val_data, test_data = get_train_test_val()
        train_numb = len(train_data)
        valid_numb = len(val_data)
    model = attention_resnet()

    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)

    with tf.device('/gpu:0'):
        model.fit_generator(generator=generate_train_batch_data(BATCH, train_data), steps_per_epoch=train_numb // BATCH, epochs=EPOCH,
                            verbose=1, validation_data=generate_val_batch_data(BATCH, val_data), validation_steps=valid_numb // BATCH,
                            callbacks=callable, max_q_size=1)


def evaluate():
    test_data, test_label = get_test_data()
    model = attention_resnet()
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
        print(i)
        test_data, test_label = get_one_label_test_data(i)
        model = attention_resnet()
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
    model = attention_resnet()
    model.load_weights(MODEL_WEIGHTS_PATH)
    preds = model.predict(test_data, batch_size=BATCH)
    np.save("16_arnet_label.npy", test_label)
    np.save("16_arnet_predict.npy", preds)

    return test_label, preds

if __name__ == "__main__":
    # train()
    # evaluate()
    # evaluate_single_class()
    predict()