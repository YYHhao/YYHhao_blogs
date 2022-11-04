# 简介
**numpy**通常与**scipy**和**matplotlib**库一起使用,它的核心是ndarray对象，支持大量的维数数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。    

SciPy包含的模块有最优化、线性代数、积分、插值、特殊函数、快速傅里叶变换、信号处理和图像处理、常微分方程求解和其他科学与工程中常用的计算。    

Matplotlib是Python编程语言及其数值数学扩展包 NumPy 的可视化操作界面。它为利用通用的图形用户界面工具包，如Tkinter, wxPython, Qt或GTK+向应用程序嵌入式绘图提供了应用程序接口（API）。

[numpy中文文档](https://www.numpy.org.cn/user/setting-up.html)   

# 基本操作   
## 求和   
**np.sum(a, axis, dtype, out, keepdims, initial, where)**    
```python
import numpy as np
i = np.array([[1,5],[3,3]])
j = np.array([[2,1],[4,4]])
print(i-j)  #对应元素相减
a = np.sum(i)
b = sum(i)
c = np.sum(i,axis=0,keepdims=True) #保留原维度，默认是False
d = np.sum(i,axis=1,keepdims=True)
print(a,end='\n\n')
print(b,end='\n\n')
print(c,end='\n\n')
print(d)
```
输出：  
```
[[-1  4]
 [-1 -1]]
12

[4 8]

[4 8]

[6 6]
```
[np.sum()参数](https://blog.csdn.net/weixin_41377182/article/details/125399751)

## 数组的堆叠和平铺   
 
**np.hstack()和np.c_[],水平平铺**   
```python
import numpy as np
#np.hstack()和np.c_[],水平平铺
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
print('水平平铺')
print('np.hstack((a, b))=\n',np.hstack((a,b)))
print('np.c_[a, b]=\n',np.c_[a,b])

#只有当a,b为一维数组时,两种函数结果不同
a1 = a.flatten()
b1 = b.flatten()
print('a1=\n',a1)
print('b1=\n',b1)
print('np.hstack((a1,b1))=\n',np.hstack((a1,b1)))
print('np.c_[a,b]=\n',np.c_[a1,b1])
```
输出：
```
水平平铺
np.hstack((a, b))=
 [[1 2 5 6]
 [3 4 7 8]]
np.c_[a, b]=
 [[1 2 5 6]
 [3 4 7 8]]
a1=
 [1 2 3 4]
b1=
 [5 6 7 8]
np.hstack((a1,b1))=
 [1 2 3 4 5 6 7 8]
np.c_[a,b]=
 [[1 5]
 [2 6]
 [3 7]
 [4 8]]
```
   
**np.vstack()和np.r_[],竖直堆叠**   
```python
import numpy as np
#np.vstack()和np.r_[],竖直堆叠
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
print('竖直堆叠')
print('np.vstack((a, b))=\n',np.vstack((a,b)))
print('np.r_[a, b]=\n',np.r_[a,b])

#只有当a,b为一维数组时,两种函数结果不同
a1 = a.flatten()
b1 = b.flatten()
print('a1=\n',a1)
print('b1=\n',b1)
print('np.vstack((a1,b1))=\n',np.vstack((a1,b1)))
print('np.r_[a,b]=\n',np.r_[a1,b1])
```
输出：
```
竖直堆叠
np.vstack((a, b))=
 [[1 2]
 [3 4]
 [5 6]
 [7 8]]
np.r_[a, b]=
 [[1 2]
 [3 4]
 [5 6]
 [7 8]]
a1=
 [1 2 3 4]
b1=
 [5 6 7 8]
np.vstack((a1,b1))=
 [[1 2 3 4]
 [5 6 7 8]]
np.r_[a,b]=
 [1 2 3 4 5 6 7 8]
```

**np.concatenate()是对array进行拼接的函数，也能实现上述过程**   
```python
import numpy as np
np.random.seed(1)
x1 = np.random.normal(0,1,(2,3)) #随机生成均值为0，标准差为1，size为(2,3)的随机数组
x2 = np.random.normal(1,2,(2,3))
print('x1=\n',x1)
print('x2=\n',x2)
print(np.concatenate([x1,x2],axis=1)) #按水平方向
print(np.concatenate([x1,x2],axis=0)) #按竖直反向
```
输出：
```
x1=
 [[ 1.62434536 -0.61175641 -0.52817175]
 [-1.07296862  0.86540763 -2.3015387 ]]
x2=
 [[ 4.48962353 -0.5224138   1.63807819]
 [ 0.50125925  3.92421587 -3.12028142]]
  
[[ 1.62434536 -0.61175641 -0.52817175  4.48962353 -0.5224138   1.63807819]
 [-1.07296862  0.86540763 -2.3015387   0.50125925  3.92421587 -3.12028142]]

[[ 1.62434536 -0.61175641 -0.52817175]
 [-1.07296862  0.86540763 -2.3015387 ]
 [ 4.48962353 -0.5224138   1.63807819]
 [ 0.50125925  3.92421587 -3.12028142]]
```