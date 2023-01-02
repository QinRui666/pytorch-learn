# -*- coding:utf-8 -*-
"""
作者：QinRui
日期：2023年01月02日
"""
import numpy as np

import  imageio
import torch
#
# img_arr=imageio.imread('/home/user1/QinRui/learn-dataset/dog.jpg')
# print(img_arr.shape)
# img=torch.from_numpy(img_arr)
# out=img.permute(2,0,1)
# print(out.shape)

# batch_size=3
# batch=torch.zeros(batch_size,3,256,256,dtype=torch.uint8) # 批处理将由3幅高度256像素、宽度256像素的RGB图像组成。
# 注意张量的类型：期望每种颜色都以8位整数表示
'''
从一个输入目录中加载所有的PNG图像，并将它们存储在张量中
'''
# import os
# data_dir='/home/user1/QinRui/learn-dataset/image-cats'
# filenames=[name for name in os.listdir(data_dir)
#            if os.path.splitext(name)[-1]=='.png']
# for i,filename in enumerate(filenames):
#     img_arr=imageio.imread(os.path.join(data_dir,filename))
#     img_t=torch.from_numpy(img_arr)
#     img_t=img_t.permute(2,0,1)
#     img_t=img_t[:3]# 只保留前3个通道，有时图像还有一个表示透明度的alpha通道，但我们只需要RGB输入
#     batch[i]=img_t
#
# print(batch.shape)


'''
加载文件并将生成的numpy数组转换为pytorch张量
'''
import csv
wine_path="/home/user1/QinRui/learn-dataset/winequality-white.csv"
wine_numpy=np.loadtxt(wine_path,dtype=np.float32,delimiter=";",skiprows=1)
print(wine_numpy)
col_list=next(csv.reader(open(wine_path),delimiter=';'))
wine=torch.from_numpy(wine_numpy)# 得到一个浮点数的torch.Tensor对象，它包含所有列。

data=wine[:,:-1]# 选择所有行和除最后一列以外的所有列
target=wine[:,-1]# 选择所有行和最后一列
'''
2种方法将target张量转换为标签张量
'''
target=wine[:,-1].long()#简单地将标签视为分数的整数向量
### 另一种方法是构建独热编码（one-hot encoding），即将10个分数分别编码到一个由10个元素组成的向量中。
target_onehot=torch.zeros(target.shape[0],10)
target_onehot.scatter_(1,target.unsqueeze(1),1.0)# 使用scatter_()方法获得一个独热编码。