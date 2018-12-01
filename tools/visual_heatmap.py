"""

@author: zhiweihu
@create time: 2018-11-17 12:29
@site:https://bindog.github.io/blog/2018/02/10/model-explanation/
@site:https://blog.csdn.net/weiwei9363/article/details/79112872
@site:https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb

"""
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras import backend as K
K.set_learning_phase(0)

model_path = "/home/science/huzhiwei/disease/model/16_attention_resnet.hdf5"
input_path = "/home/science/huzhiwei/disease/paper/input"
prefix_path = "/home/science/huzhiwei/disease/paper"

# 为了解决RuntimeError: Invalid DISPLAY variable异常
plt.switch_backend('agg')

def standartize(array):
    array = np.array(array, dtype=np.float32)
    mean = np.mean(array)
    std = np.std(array)
    array -= mean
    array /= std
    return array


def visual_heatmap(input_path, model_path, prefix_path):
    file_list = os.listdir(input_path)
    for i, file_name in enumerate(file_list):
        image_path = input_path + "/" + file_name
        img = load_img(image_path)
        img = img_to_array(img)
        img = standartize(img)
        img_tensor = np.expand_dims(img, axis=0)

        model = load_model(model_path)
        model.summary()

        preds = model.predict(img_tensor)
        class_num = np.argmax(preds[0])

        leaf_output = model.output[:, class_num]
        # conv_layer = model.get_layer('add_32')
        # conv_layer = model.get_layer('add_29')
        conv_layer = model.get_layer('add_19')
        grads = K.gradients(leaf_output, conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.layers[0].input], [pooled_grads, conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([img_tensor])
        # for i in range(2048):
        # for i in range(1024):
        # for i in range(512):
        for i in range(256):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        # heatmap = np.minimum(heatmap, 0)
        # heatmap /= np.min(heatmap)
        plt.matshow(heatmap)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(prefix_path + '/feature/' + file_name[0: file_name.index(".jpg")] + ".png")
        # plt.show()

        img = cv2.imread(image_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img
        cv2.imwrite(prefix_path + '/output/' + file_name, superimposed_img)


if __name__ == "__main__":
    visual_heatmap(input_path=input_path, model_path=model_path, prefix_path=prefix_path)
