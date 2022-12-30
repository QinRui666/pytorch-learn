# -*- coding:utf-8 -*-
"""
作者：QinRui
日期：2022年12月22日
"""

from torchvision import models
# print(dir(models)) #输出已有的模型列表
alexnet=models.AlexNet()
resnet=models.resnet101(pretrained=True)
from torchvision import transforms
preprocess=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
from PIL import Image
img=Image.open("/home/user1/QinRui/learn-dataset/dog.jpg")
img_t=preprocess(img)
import torch
batch_t=torch.unsqueeze(img_t,0)

resnet.eval()
out=resnet(batch_t)

# 加载一个包含1000个标签的文件
with open('/home/user1/QinRui/learn-dataset/imagenet_classes.txt') as f:
    labels=[line.strip() for line in f.readlines()]

# 确定out张量中最高得分对应的索引
_, index=torch.max(out,1)

# 使用index[0]获得实际的数字作为标签列表的索引，并用softmax使输出归一化
percentage=torch.nn.functional.softmax(out,dim=1)[0]*100
print(labels[index[0]],percentage[index[0]].item())
