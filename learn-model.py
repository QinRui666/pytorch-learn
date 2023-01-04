# -*- coding:utf-8 -*-
"""
作者：QinRui
日期：2023年01月03日
"""
import  torch
'''
得到以下数据
'''
t_c=[0.5,14.0,15.0,28.0,11.0,8.0,3.0,-4.0,6.0,13.0,21.0]
t_u=[35.7,55.9,58.2,81.9,56.3,48.9,33.9,21.8,48.4,60.4,68.4]
t_c=torch.tensor(t_c)# t_c值是以摄氏度为单位的温度
t_u=torch.tensor(t_u)# t_u值是我们未知的单位

'''
实现模型和损失函数
'''
def model(t_u,w,b): #分别作为输入张量、权重参数和偏置参数
    return w*t_u+b

# 构建一个差分张量，先对其平方元素进行处理，最后通过对得到的张量中的所有元素求平均值得到一个标量损失函数，即均方损失函数
def loss_fn(t_p,t_c):
    squared_diffs=(t_p-t_c)**2
    return squared_diffs.mean()

'''
初始化参数,调用模型并检查损失的值
'''
w=torch.ones(())
b=torch.zeros(())

t_p=model(t_u,w,b)
print(t_p)

loss=loss_fn(t_p,t_c)
print(loss)

'''
在w和b的当前值附近，w的增加会导致损失的一些变化。
如果变化是负的，那么我们需要增加w来最小化损失，而如果变化是正的，我们需要减小w的值。
那么值具体增加或减少多少呢？
我们通常用一个小的比例因子来衡量变化率，在机器学习中称为学习率（learning_rate）
'''
delta=0.1
loss_rate_of_change_w=(loss_fn(model(t_u,w+delta,b),t_c)-loss_fn(model(t_u,w-delta,b),t_c))/(2.0*delta)

learning_rate=1e-2
w=w-learning_rate*loss_rate_of_change_w
loss_rate_of_change_b=(loss_fn(model(t_u,w,b+delta),t_c)-loss_fn(model(t_u,w,b-delta),t_c))/(2.0*delta)
b=b-learning_rate*loss_rate_of_change_b


'''
计算导数，应用链式法则，先计算损失对于其输入(模型的输出)的导数，再乘模型对参数的导数
'''
def loss_fn(t_p,t_c):
    squared_diffs=(t_p-t_c)**2
    return squared_diffs.mean()
def dloss_fn(t_p,t_c):
    dsq_diffs=2*(t_p-t_c)/t_p.size(0)
    return dsq_diffs
'''
将导数应用到模型中
'''
def model(t_u,w,b):
    return w*t_u+b
def dmodel_dw(t_u,w,b):
    return t_u
def dmodel_db(t_u,w,b):
    return 1.0
'''
定义梯度函数
'''
def grad_fn(t_u,t_c,t_p,w,b):
    dloss_dtp=dloss_fn(t_p,t_c)
    dloss_dw=dloss_dtp*dmodel_dw(t_u,w,b)
    dloss_db=dloss_dtp*dmodel_db(t_u,w,b)
    return torch.stack([dloss_dw.sum(),dloss_db.sum()])


'''
循环训练
'''
# def training_loop(n_epochs,learning_rate,params,t_u,t_c):
#     for epoch in range(1,n_epochs+1):
#         w,b=params
#         t_p=model(t_u,w,b) # 正向传播
#         loss=loss_fn(t_p,t_c)
#         grad=grad_fn(t_u,t_c,t_p,w,b)# 反向传播
#         params=params-learning_rate*grad
#         print('Epoch %d, Loss %f' % (epoch,float(loss)))
#
#     return params


'''
调用循环训练
'''
# params=training_loop(
#     n_epochs=5000,
#     learning_rate=1e-2,
#     params=torch.tensor([1.0,0.0]),
#     t_u=t_u*0.1,
#     t_c=t_c,
# )

'''
可视化数据
'''

# from matplotlib import pyplot as plt
# t_p=model(t_u*0.1,*params)
# fig=plt.figure(dpi=600)
# plt.xlabel("Temperature(°Fahrenheit)")
# plt.ylabel("Temperature(°Celsius)")
# plt.plot(t_u.numpy(),t_p.detach().numpy())
# plt.plot(t_u.numpy(),t_c.numpy(),'o')
# plt.savefig("temp_unknown_plot.png", format="png")
# plt.show()

'''
应用自动求导
'''
params=torch.tensor([1.0,0.0],requires_grad=True)# 再一次初始化一个参数张量
# requires_grad=True使任何将params作为祖先的张量都可以访问从params到那个张量调用的函数链。
# 导数的值将自动填充为params张量的gard属性
'''
使用gard属性
'''
# 我们要做的是从一个requires_gard=Ture的张量开始，调用模型并计算损失，然后反向调用损失张量
loss=loss_fn(model(t_u,*params),t_c)
loss.backward()
params.grad# 此时，params的gard属性包含关于params的每个元素的损失的导数

'''
累加梯度函数
'''
# 调用backward()将导致导数在叶节点上累加，因此如果提前调用backward()，则会再次计算损失
# 再次调用backward()，每个叶节点上的梯度将在上一次迭代中计算的梯度上累加，这会导致梯度计算不正确
if params.grad is not None:
   params.grad.zero_() # 每次迭代时明确将梯度归零

'''
自动求导训练代码
'''
# def training_loop(n_epochs,learning_rate,params,t_u,t_c):
#     for epoch in range(1,n_epochs+1):
#         if params.grad is not None:
#             params.grad.zero_()
#
#         t_p=model(t_u,*params)
#         loss=loss_fn(t_p,t_c)
#         loss.backward()
#
#         with torch.no_grad():
#             params-=learning_rate*params.grad # 使用with语句将更新封装在非梯度上下文中，意味着在with块中，自动求导机制不起作用
#
#         if epoch % 500==0:
#             print('Epoch %d, Loss %f'% (epoch,float(loss)))
#
#     return params
#
# training_loop(
#     n_epochs=5000,
#     learning_rate=1e-2,
#     params=torch.tensor([1.0,0.0],requires_grad=True),
#     t_u=0.1*t_u,
#     t_c=t_c
# )

'''
使用一个梯度下降优化器
'''
import torch.optim as optim
params=torch.tensor([1.0,0.0],requires_grad=True)
learning_rate=1e-2
optimizer=optim.SGD([params],lr=learning_rate) # SGD代表随机梯度下降

'''
更新训练循环
'''
def training_loop(n_epochs,optimizer,params,t_u,t_c):
    for epoch in range(1,n_epochs+1):

        t_p=model(t_u,*params)
        loss=loss_fn(t_p,t_c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # params的值在调用step()时更新，即优化器会查看params.gard并更新params,从中减去学习率乘梯度

        if epoch % 500==0:
            print('Epoch %d, Loss %f'% (epoch,float(loss)))

    return params

training_loop(
    n_epochs=5000,
    optimizer=optimizer,
    params=params,
    t_u=t_u*0.1,
    t_c=t_c
)
print(params)