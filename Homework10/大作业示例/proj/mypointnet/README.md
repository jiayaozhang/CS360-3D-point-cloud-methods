# MyPointNet

## 环境配置
- 安装pytorch
- 安装tensorboard
- 安装numpy
- 安装pretty-errors

## 文件结构介绍
- output
  - latest.pth：准确率最好的一次权重参数结果
  - traced_model.pt：最好的一次模型和参数结果，用于C++接口调用
- src
  - model.py：模型代码
  - train.py：训练代码
  - test.py：测试代码
  - dataset.py：数据集处理代码

## 训练方法
- 注意调整绚练和测试代码中数据集类输入的数据集根目录地址
```
cd ${代码根目录}
tensorboard --logdir output/runs/tersorboard
```
> 打开浏览器，输入网址：http://localhost:6006/
```
cd ${代码根目录}/src
python train.py
```

## 测试方法
```
cd ${代码根目录}/src
python test.py
```