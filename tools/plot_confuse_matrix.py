"""

@author: zhiweihu
@create time: 2018-11-21 11:32
@site:热力图可视化工具：seaborn

"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

num_classes = 10

plt.switch_backend('agg')

def plot_confuse_matrix(label_path, predict_path, save_path, title):
    """
    绘制混淆矩阵(注意混淆矩阵输入的数据不是one-hot类型)
    :param label_path:
    :param predict_path:
    :param save_path:
    :return:
    """
    # 将坐标轴字体设置为Time New Roman
    plt.rc('font', family='Times New Roman')
    pre_label = np.load(label_path)
    label = np.argmax(pre_label, axis=1)
    numclass_count = np.zeros(num_classes, dtype=np.int32)
    for i in range(pre_label.shape[0]):
        index = label[i]
        numclass_count[index] = numclass_count[index] + 1
    pre_predict = np.load(predict_path)
    predict = np.argmax(pre_predict, axis=1)
    confuse_matrix = confusion_matrix(label, predict)
    df_cm = pd.DataFrame(confuse_matrix)
    f, ax = plt.subplots(figsize=(9, 6))
    # 将横轴坐标显示在最上面
    ax.xaxis.tick_bottom()
    # 设置坐标轴字体大小
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    # 绘制热力图，还要将数值写到热力图上
    sns.heatmap(df_cm, annot=True, fmt="d", ax=ax,  annot_kws={'size': 15, 'weight': 'bold'}, cmap=plt.get_cmap('PuBuGn'))
    # 设置坐标字体方向
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=360, horizontalalignment='right')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    # 绘制ROC曲线
    # label = np.load(label_path)
    # predict = np.load(predict_path)
    # for i in range(10):
    #     plot_single_class_roc(label, predict, class_num=i)


    # 绘制混淆矩阵
    label_path = "F:/python_workspace/agriculture/diease/finally/predict/16_arnet_label.npy"
    predict_path = "F:/python_workspace/agriculture/diease/finally/predict/16_arnet_predict.npy"
    save_path = "F:/python_workspace/agriculture/diease/finally/predict/arnet.png"
    title = "ARNet"
    plot_confuse_matrix(label_path, predict_path, save_path, title)