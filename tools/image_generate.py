"""

@author: zhiweihu
@create time: 2018-11-9 16:02

"""
import numpy as np
import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def disease_class_count(image_path, directory_name):
    """
    计算每种类别下叶片的数量
    :param image_path:
    :return:
    """
    count = 0
    one_disease_directory_path = image_path + '/' + directory_name
    one_disease_directory_file_list = os.listdir(one_disease_directory_path)
    for j, file_name in enumerate(one_disease_directory_file_list):
        count = count + 1
      # print("now process %s, count %d" % (directory_name, count))
    return count


def data_generate(input_path, aug_size):
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
    file_list = os.listdir(input_path)
    for i, directory_name in enumerate(file_list):
        print("now process %s " % (directory_name))
        one_disease_directory_path = input_path + '/' + directory_name
        one_disease_directory_file_list = os.listdir(one_disease_directory_path)
        start_count = disease_class_count(input_path, directory_name)
        count = start_count
        for j, file_name in enumerate(one_disease_directory_file_list):
            image = load_img(one_disease_directory_path + "/" + file_name)
            image = img_to_array(image)
            # 扩充一个维度
            image = np.expand_dims(image, axis=0)
            # 生成图片
            gen = aug.flow(image, batch_size=1)

            # 保存生成的图片
            for i in range(aug_size):
                x_batch = next(gen)
                result_image = x_batch[0]
                result_image = array_to_img(result_image)
                if not os.path.lexists(input_path + '/' + directory_name):
                    os.mkdir(input_path + '/' + directory_name)
                result_image.save(input_path + '/' + directory_name + "/" + str(count) + ".jpg")
                count = count + 1


if __name__ == "__main__":
    data_generate("F:/python_workspace/agriculture/diease/paper/", 4)

