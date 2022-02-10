项目简单介绍

本项目实现了端到端的实体文字检测，识别和筛输出处理，可以满足大部分的企业级实体的文字识别。

环境安装



`pip3 install -r requirments.txt`



```
pip3 install --upgrade pip

如果您的机器安装的是CUDA9，请运行以下命令安装
python3 -m pip install paddlepaddle-gpu==1.7.2.post97 -i https://pypi.tuna.tsinghua.edu.cn/simple

如果您的机器安装的是CUDA10，请运行以下命令安装
python3 -m pip install paddlepaddle-gpu==1.7.2.post107 -i https://pypi.tuna.tsinghua.edu.cn/simple

如果您的机器是CPU，请运行以下命令安装

python3 -m pip install paddlepaddle==1.7.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
 
NLP依赖包
pip install transformers
```


快速开始



~~~~
python main.py --image_dir ./data/imgs_CN/ --use_angle_cls true --cls true
~~~~
自定义图片数据集
~~~~
改变--image_dir /your_dir/
~~~~
功能介绍
****
1.有效的图片上的字符进行检测和识别

2.对于关键招牌文字的输出

3.对于内容信息较为多的图片也可以很好的输出结果

4.输出格式为地址+招牌名（保存于data.xls）


****
完善中的功能

1.通过先进行实例分割对图片进行处理，来加快后面运算的速度

2.兼容多种模型的使用
