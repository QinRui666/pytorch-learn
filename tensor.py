# -*- coding:utf-8 -*-
"""
作者：QinRui
日期：2023年01月01日
"""
import torch      # 导入模块
# a=torch.ones(3)   # 创建一个大小为3的一维张量，用1.0填充
# print(a)
# print(a[1])
# print(float(a[1]))
# a[2]=2.0
# print(a)
'''
我们可以使用一维张量，将x轴坐标存储在偶数索引中，将y轴坐标存储在奇数索引中
'''
p=torch.zeros(6)# 使用zeros()函数只是获得适当大小的数组的一种方法
p[0]=4.0
p[1]=1.0
p[2]=5.0
p[3]=3.0
p[4]=2.0
p[5]=1.0
'''
还可以向构造函数传递一个Python列表达到同样的效果
'''
pp=torch.tensor([4.0,1.0,5.0,3.0,2.0,1.0])
# print((float(p[0]),float(p[1]))) #输出第一个点的坐标
'''
尽管将第1个索引指向单独的二维点而不是点坐标是可行的，但对于这种情况，我们可以用一个二维张量
'''
ppp=torch.tensor([[4.0,1.0],[5.0,3.0],[2.0,1.0]])
# print(ppp)
# print(ppp.shape) # 查看张量形状
# print(ppp[0,1])# 可以用2个索引来访问张量中的单个元素
# print(ppp[0]) # 也可以访问张量中的第1个元素，得到第1个点的二维坐标

'''
列表中元素索引
'''
some_list=list(range(6))
some_list[:]  #列表中所有元素
some_list[1:4]# 包含第1个元素到第3个元素，不包含第4个元素
some_list[1:] # 包含第1个元素到列表末尾元素
some_list[:4]# 从列表开始到第3个元素，不包含第4个元素
some_list[:-1]# 从列表开始到最后一个元素之前的所有元素
some_list[1:4:2]# 从第1个元素（包含）到第4个元素（不包含），移动步长为2

# p[1:]# 第1行之后得所有行，隐含所有列
# p[1:,:]# 第1行之后得所有行，所有列
# p[1:,0]#第1行之后得所有行，第1列
# p[None]# 增加大小为1的维度，就像unsqueeze()方法一样

'''
为了给张量分配一个正确的数字类型，我们可以指定适当的dtype作为构造函数的参数
'''
double_points=torch.ones(10,2,dtype=torch.double)
short_points=torch.tensor([[1,2],[3,4]],dtype=torch.short)
short_points.dtype# 通过访问相应的属性来了解一个张量的dtype值
double_points=torch.zeros(10,2).double()# 可以使用相应的转换方法将张量创建函数的输出转换为正确的类型
short_points=torch.ones(10,2).short()
double_points=torch.zeros(10,2).to(torch.double) #或者用更方便的方法

points=torch.tensor([[4.0,1.0],[5.0,3.0],[2.0,1.0]])
# print(points.storage())
points_storage=points.storage()
points_storage[0]
points.storage()[1]

second_point=points[1]
second_point.storage_offset() #得到的张量在存储区中的偏移量为2，这是因为我们需要跳过第1个点，该点有两个元素
second_point.size()
second_point.shape

second_point[0]=10.0
print(points)# 更改子张量会对原始张量产生影响
second_point=points[1].clone()# 可以把这子张量克隆成一个新的张量
second_point[0]=10.0


'''
转置不会分配新的内存，只是创建一个新Tensor实例，该实例具有与原始张量不同的步长顺序
'''
points_t=points.t() #t()是用于二维张量转置的transpose()方法的简写
id(points.storage())==id(points_t.storage())# 可以验证2个张量共享同一个存储区
points.stride()
points_t.stride() #它们两在形状和步长上不一置

points_gpu=torch.tensor([[4.0,1.0],[5.0,3.0],[2.0,1.0]],device='cuda')# 在gpu上创建了一个张量
points_gpu=points.to(device='cuda')# 用to()方法将在cpu上创建的张量复制到gpu上
points_gpu=points.to(device='cuda:0')#确定多个gpu下存储张量的gpu

'''
从张量到数组
'''
points=torch.ones(3,4)
points_np=points.numpy()# 返回一个大小、形状和数字类型都与代码对应的NumPy多维数组
# 如果张量是在gpu上存储的，PyTorch将把张量的内容复制到cpu上分配的numpy数组中
'''
从数组到张量
'''
points=torch.from_numpy(points_np)
