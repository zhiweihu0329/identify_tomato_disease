"""

@author: zhiweihu
@create time: 2018-11-7 21:17

"""

import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import *
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle


IMAGE_PATH = "/home/science/huzhiwei/disease/real_datasets"
MODEL_WEIGHTS_PATH = "/home/science/huzhiwei/disease/model/16_vgg16.hdf5"

LEARNING_RATE = 1e-3
BATCH = 16
EPOCH = 100
num_classes = 10

# 为了解决RuntimeError: Invalid DISPLAY variable异常
plt.switch_backend('agg')


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


def vgg16():
    """
    用预训练权重参数构建vgg16模型
    :return:
    """
    input_tensor = Input(shape=(256, 256, 3))
    base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    x = Flatten()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # freeze all base_model layers
    for layer in base_model.layers:
        layer.trainable = False

    return model

def plot_single_class_roc(label, predict, class_num=0):
    """
    根据预测结果绘制指定类的roc曲线图
    :param label: 真实标注值
    :param predict: 预测值
    :param class_num: 需要绘制的类(从0开始)
    :return:
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(label[:, i], predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(label.ravel(), predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    plt.plot(fpr[class_num], tpr[class_num], color='darkorange',
             lw=class_num, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=class_num, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("1.jpg")

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=class_num,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=class_num)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig("2.jpg")


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


def get_all_predict():
    """
    获取所有测试集上的label以及其预测结果predict
    :return:
    """
    test_data, test_label = get_test_data()
    model = vgg16()
    model.load_weights(MODEL_WEIGHTS_PATH)
    preds = model.predict(test_data)

    return test_label, preds


def start_predict():
    label, predict = get_all_predict()
    plot_single_class_roc(label, predict, class_num=0)


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
        model_checkpoint = ModelCheckpoint(str(BATCH) + '_vgg16.hdf5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
        # 自动变化学习率
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1, patience=5, min_lr=0.0001)
        callable = [model_checkpoint, early_stopping, reduce_lr]
        train_data, val_data, test_data = get_train_test_val()
        train_numb = len(train_data)
        valid_numb = len(val_data)
    model = vgg16()

    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)

    with tf.device('/gpu:0'):
        model.fit_generator(generator=generate_train_batch_data(BATCH, train_data), steps_per_epoch=train_numb // BATCH, epochs=EPOCH,
                            verbose=1, validation_data=generate_val_batch_data(BATCH, val_data), validation_steps=valid_numb // BATCH,
                            callbacks=callable, max_q_size=1)

def evaluate():
    test_data, test_label = get_test_data()
    model = vgg16()
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
        model = vgg16()
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
    model = vgg16()
    model.load_weights(MODEL_WEIGHTS_PATH)
    preds = model.predict(test_data, batch_size=BATCH)
    np.save("16_vgg16_label.npy", test_label)
    np.save("16_vgg16_predict.npy", preds)

    return test_label, preds

if __name__ == "__main__":
    # train()
    # start_predict()
    # evaluate()
    # evaluate_single_class()
    predict()