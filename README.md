# identify_tomato_disease
##### 本代码库是论文《基于注意力残差机制的细粒度番茄病害识别》的实现，基于keras框架完成
##### models/路径下分别是文章中提出了ARNet模型及其baseline，各模型注释如下：
* attention_resnet.py：ARNet模型
* inceptionv3.py：InceptionV3模型
* mobilenetv2.py：MobileNetV2模型
* resnet34.py：ResNet34模型
* vgg16.py：VGG16模型
* xception.py：Xception模型
##### tools/路径下均为数据预处理以及论文图表绘制代码。注释如下：
* image_generate.py：对训练集图片进行数据增强操作
* plot_confuse_matrix.py：将预测结果绘制为混淆矩阵
* plot_roc_auc.py：将预测结果绘制为ROC曲线并计算AUC值
* preprocess.py：预处理图片
* visual_convolution.py：可视化模型中间卷积层输出
* visual_heatmap.py：热力图绘制
