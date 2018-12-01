"""

@author: zhiweihu
@create time: 2018-12-1 10:20

"""
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras import models

layer_list = ["my_conv1", "add_17", "add_20", "add_24", "add_30"]
model_path = "/home/science/huzhiwei/disease/model/16_attention_resnet.hdf5"
input_path = "/home/science/huzhiwei/disease/paper/input"
prefix_path = "/home/science/huzhiwei/disease/paper"
save_image_path = "F:/python_workspace/agriculture/diease/paper/中间层可视化"

def standartize(array):
    array = np.array(array, dtype=np.float32)
    mean = np.mean(array)
    std = np.std(array)
    array -= mean
    array /= std
    return array


def visual_convolution(image_path, model_path, save_image_path):
    img = load_img(image_path)
    img = img_to_array(img)
    img = standartize(img)
    img_tensor = np.expand_dims(img, axis=0)

    model = load_model(model_path)
    model.summary()
    layer_outputs = []
    for i in range(len(layer_list)):
        layer_outputs.append(model.get_layer(layer_list[i]).output)
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    for j in range(len(layer_list)):
        first_layer_activation = activations[j]

        # [0, :, :, 3]第一个参数0表示预测结果中的第几个,第三个参数3表示第几个通道
        channel_nums = first_layer_activation.shape[3]
        count = 0
        for k in range(channel_nums):
            count = count + 1
            plt.matshow(first_layer_activation[0, :, :, k], cmap='viridis')
            plt.axis('off')
            plt.xticks([]) #去掉坐标轴刻度
            plt.yticks([])
            if not os.path.lexists(save_image_path + '/' + layer_list[j]):
                os.mkdir(save_image_path + '/' + layer_list[j])
            plt.savefig(save_image_path + '/' + layer_list[j] + "/" + str(count) + ".png")
            # plt.show()


if __name__ == "__main__":
    visual_convolution(image_path=image_path, model_path=model_path, save_image_path=save_image_path)