# 损失函数的定义 
损失函数，又叫目标函数，是编译一个神经网络模型必须的两个参数之一，另一个是优化器。其本质就是计算标签值(真实值)和预测值之间的差异的函数。  
# 损失函数的用途    
机器学习中有多种损失函数可供选择，我们希望预测值无限接近真实值，所以需要将差值降到最低（在这个过程中就需要引入损失函数）。而在此过程中损失函数的选择是十分关键的，在具体的项目中，有些损失函数计算的差值梯度下降的快，而有些下降的慢，所以选择合适的损失函数也是十分关键的。  
# 常见的损失函数(基于pytorch)  
首先定义两个二维数组，然后用不同损失函数计算其损失值。       
```python
from random import sample
from torch.autograd import Variable
import torch 
import torch.nn as nn

sample = Variable(torch.ones(2,2))
a = torch.Tensor(2,2)
a[0,0] = 0
a[0,1] = 1
a[1,0] = 2
a[1,1] = 3
target = Variable(a)
print(sample)
print(target)
```  
输出：
```
tensor([[1., 1.],
        [1., 1.]])
tensor([[0., 1.],
        [2., 3.]])
```

## 1、L1Loss   
L1Loss就是取预测值和真实值的绝对误差的平均数。  
公式：

$$loss(x,y)=\frac{1}{N}\sum_{i=1}^N|x-y|$$  
```python
criterion = nn.L1Loss()   #尺度，准则
loss = criterion(sample,target)
print(loss)  #1
```  
## 2、SmoothL1Loss
SmoothL1Loss也称为Huber Loss，误差在(-1,1)上是平方损失，其它情况是L1损失。  
公式：

$$loss(x,y)=\frac{1}{N}\begin{cases} 
\frac{1}{2}(x_i-y_i)^2, & if|x_i-y_i|<1 \\
|x_i-y_i|-\frac{1}{2}, & otherwise\\
\end{cases}
$$
```python
criterion = nn.SmoothL1Loss()   
loss = criterion(sample,target)
print(loss)  #0.625
```    
## 3、MSELoss(均方误差)  
计算预测值和真实值之间的平方和的平均数。  
公式：

$$loss(x,y)=\frac{1}{N}\sum_{i=1}^{N}|x-y|^2$$  
```python
criterion = nn.MSELoss()   
loss = criterion(sample,target)
print(loss)  #1.5
```  
## 4、BCELoss
二分类用的交叉熵，计算公式较为复杂。  
公式：

$$loss(o,t)=-\frac{1}{N}\sum_{i=1}^{N}[t_i*log(o_i)+(1-t_i)*log(1-o_i)]$$  
```python
criterion = nn.BCELoss()   
loss = criterion(sample,target)
print(loss)  #-13.8155
```  
## 5、CrossEntropyLoss  
交叉熵损失函数，在图像分类神经网络模型中常用该公式。  
公式：

$$loss(x,label)=-log\frac{e^{x_{label}}}{\sum_{j=1}^{N}e^{x_{j}}} =-x_{label}+log\sum_{j=1}^{N}e^{x_{j}}$$

## 6、NLLLoss
负对数似然损失函数(Negative Log Likelihood),在前面接一个**LogSoftMax**层就等价于交叉熵损失了。注意这里的x,label和上个交叉熵损失里的不同，这里是经过log运算后的数值。该损失函数一般也用到图像识别模型上。  
公式：

$$loss(x,label)=-x_{label}$$    

NLLLoss和CrossEntropyLoss的功能很相似，通常都用到多分类模型中，实际应用中NLLLoss损失函数的使用更多。 

## 7、NLLLoss2d
与上述损失函数类似，多了几个维度，一般用到图片上。比如用全卷积网络分类时，最后图片的每个点都会预测一个类别标签。    
input,(N,C,H,W)  
target,(N,H,W)  



[softmax参考](https://blog.csdn.net/lz_peter/article/details/84574716)    
[交叉熵参考](https://blog.csdn.net/watermelon1123/article/details/91044856)  
[https://blog.csdn.net/yeldon/article/details/109020841](https://blog.csdn.net/yeldon/article/details/109020841)