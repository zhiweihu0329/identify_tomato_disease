"""

@author: zhiweihu
@create time: 2018-12-1 10:35

"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle

num_classes = 10

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
    plt.savefig(str(class_num) + ".jpg")


def plot_all_class_roc(label, predict, line_width, save_path):
    """
    根据预测结果绘制指定类的roc曲线图
    :param label: 真实标注值
    :param predict: 预测值
    :param save_path:保存路径
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
        plt.plot(fpr[i], tpr[i], color=color, lw=line_width,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=line_width)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(save_path)


if __name__ == "__main__":
    # 绘制ROC曲线
    label_path = "F:/python_workspace/agriculture/diease/finally/predict/16_arnet_label.npy"
    predict_path = "F:/python_workspace/agriculture/diease/finally/predict/16_arnet_predict.npy"
    save_path = "F:/python_workspace/agriculture/diease/finally/predict/arnet.png"
    label = np.load(label_path)
    predict = np.load(predict_path)
    for i in range(10):
        plot_single_class_roc(label, predict, class_num=i)
    plot_all_class_roc(label, predict, 1, save_path)