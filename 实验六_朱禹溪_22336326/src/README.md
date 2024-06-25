# 中药图像分类的CNN模型

## 简介
该项目实现了一个卷积神经网络（CNN），用于将中药图像分类到不同的类别中。CNN模型使用PyTorch深度学习框架进行训练和评估。


本项目通过device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')选择了CPU或者GPU的运行方式。


## 安装

1. 安装所需的依赖:
    ```bash
    pip install torch torchvision matplotlib
    ```

## 使用方法


1. 准备数据集：将图像文件组织到 "train" 和 "test" 目录中。

2. 运行主要脚本:
    ```bash
    python main.py
    ```

3. 训练过程将开始，并且在结束时，脚本将绘制训练和测试损失/准确率曲线。

## 注

1. 由于代码中使用了ImageFolder函数因而需要对test集进行预处理，手动的建立子文件夹并放入图片。

2. 如果要使用GPU，请确保您的系统支持CUDA，并且正确配置了环境变量。
