# -*- coding:utf-8 -*-
"""
作者：QinRui
日期：2023年01月06日
"""
import matplotlib.pyplot as plt
import torch

'''
导入TorchVision模块并使用datasets模块下载CIFAR-10数据
'''
from torchvision import datasets
data_path='/data1/QinRui/CIFAR-10'
cifar10=datasets.CIFAR10(data_path,train=True,download=False)# 实例化一个数据集用于训练数据，如果数据集不存在，则TorchVision将下载该数据集
cifar10_val=datasets.CIFAR10(data_path,train=False,download=False)#使用train=False,获取一个数据集用于验证数据，并在需要时再次下载该数据集

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
'''
当Python对象配备了__len__()函数时，我们可以将其作为参数传递给Python的内置函数len()
'''
# print(len(cifar10))# 50000

'''
由于数据集配备了__getitem__()函数，我们可以使用标准索引对元组和列表进行索引
以访问单个数据项。这里，我们得到一个带有期望输出的PIL图像，即输出值为整数1，对应图像数据集中的”汽车“
'''
img,label=cifar10[99]
# print(img,label,class_names[label])
# plt.imshow(img)
# plt.show()

from torchvision import transforms

to_tensor=transforms.ToTensor()
img_t=to_tensor(img)
#print(img_t,'\n',img_t.shape)# 图像已变换为3*32*32的张量

'''
我们可以将变换直接作为参数传递给dataset.CIFAR10,
此时，访问数据集的元素将返回一个张量，而不是PIL图像 
'''
tensor_cifar10=datasets.CIFAR10(data_path,train=True,download=False,transform=transforms.ToTensor())
img_t,_=tensor_cifar10[99]
# print(type(img_t),img_t.shape,img_t.dtype)


'''
现在让我们计算CIFAR-10训练集的平均值和标准差。
让我们将数据集返回的所有张量沿着一个额外的维度进行堆叠
'''
imgs=torch.stack([img_t for img_t, _ in tensor_cifar10],dim=3)
print(imgs.shape)
'''
计算出每个信道的平均值
'''
imgs.view(3,-1).mean(dim=1)
# 保留3个通道，并将剩余的所有维度合并为一个维度，从而计算出适当的尺寸大小。
# 这里3*32*32的图像被转换成一个3*1024的向量，然后对每个通道的1024个元素取平均值
'''
计算标准差
'''
imgs.view(3,-1).std(dim=1)

'''
连接到ToTensor变换
'''
transformed_cifar10=datasets.CIFAR10(
    data_path,train=True,download=False,
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4915,0.4823,0.4468),
                (0.2470,0.2435,0.2616))
        ]))


img_t,_=transformed_cifar10[99]
plt.imshow(img_t.permute(1,2,0))
plt.show()