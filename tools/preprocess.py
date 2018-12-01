"""

@author: zhiweihu
@create time: 2018-11-3 8:43

"""
import json
import os
import numpy as np

from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image

RATIO = 1.0
IMAGE_SHAPE = [256, 256]

def get_tomato_image(json_path, input_image_path, output_image_path):
    """
    将json文件中提取与番茄有关的图片，其disease_class值范围为41到60之间
    :return:
    """
    json_file = open(json_path, encoding='utf-8')
    json_content = json.load(json_file)
    json_length = len(json_content)
    for i in range(json_length):
        disease_class = json_content[i]['disease_class']
        image_id = json_content[i]['image_id']
        if disease_class >= 41:
            if not os.path.lexists(output_image_path + "/" + str(disease_class)):
                os.mkdir(output_image_path + "/" + str(disease_class))
            # 将对应类别的图片存储到对应文件夹下
            pre_image_path = input_image_path + "/" + image_id
            after_image_path = output_image_path + "/" + str(disease_class) + "/" + image_id
            pre_image = load_img(pre_image_path)
            pre_image.save(after_image_path)


def preprocess_image(input_path, output_path):
    """
    预处理图片
    :param input_path:
    :return:
    """
    file_list = os.listdir(input_path)
    for i, directory_name in enumerate(file_list):
        count = 0
        print("now process %s " % (directory_name))
        one_disease_directory_path = input_path + '/' + directory_name
        one_disease_directory_file_list = os.listdir(one_disease_directory_path)
        for j, file_name in enumerate(one_disease_directory_file_list):
            count = count + 1
            image = load_img(one_disease_directory_path + "/" + file_name)
            image = img_to_array(image)
            shape_x = image.shape[0]
            shape_y = image.shape[1]
            if shape_x > shape_y:
                image_size = shape_x
            else:
                image_size = shape_y
            reshape_image = add_padding(one_disease_directory_path + "/" + file_name, [image_size, image_size])
            reshape_image = array_to_img(reshape_image)
            reshape_image = reshape_image.resize(IMAGE_SHAPE)
            if not os.path.lexists(output_path + '/' + directory_name):
                os.mkdir(output_path + '/' + directory_name)
            reshape_image.save(output_path + '/' + directory_name + "/" + str(count) + ".jpg")


def add_padding(img_path, image_size):
    """
    将指定路径图片转换为指定大小
    :param img_path:
    :param image_size:
    :return:
    """
    image = Image.open(img_path)
    w, h = image.size
    ratio = float(w) / h

    if abs(ratio - RATIO) <= 0.1:
        return np.array(image.resize(image_size))

    np_image = np.array(image)
    h, w, c = np_image.shape

    if ratio > RATIO:
        new_h = int(float(w) / RATIO)
        padding = int((new_h - h) / 2.0)

        np_new_image = np.zeros([new_h, w, c])+255
        np_new_image[padding: padding + h, :, :] = np_image

    else:
        new_w = int(float(h) * RATIO)
        padding = int((new_w - w) / 2.0)

        np_new_image = np.zeros([h, new_w, c])+255
        np_new_image[:, padding: padding + w, :] = np_image

    new_image = Image.fromarray(np.cast['uint8'](np_new_image))
    return np.array(new_image.resize(image_size))


def disease_class_count(image_path):
    """
    计算每种类别下叶片的数量
    :param image_path:
    :return:
    """
    file_list = os.listdir(image_path)
    for i, directory_name in enumerate(file_list):
        count = 0
        one_disease_directory_path = image_path + '/' + directory_name
        one_disease_directory_file_list = os.listdir(one_disease_directory_path)
        for j, file_name in enumerate(one_disease_directory_file_list):
            count = count + 1
        print("now process %s, count %d" % (directory_name, count))


if __name__ == "__main__":
    json_path = "F:/学术资料/数据集/农作物病害检测/ai_challenger_pdr2018_validation_annotations_20181021.json"
    input_image_path = "F:/学术资料/数据集/农作物病害检测/ai_challenger_pdr2018_validation_annotations_20181021"
    output_image_path = "F:/学术资料/数据集/农作物病害检测/tomato"
    output_path = "F:/学术资料/数据集/农作物病害检测/real_tomato"
    preprocess_image(output_image_path, output_path)
    # disease_class_count(output_path)