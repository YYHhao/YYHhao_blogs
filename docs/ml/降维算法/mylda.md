# LDA(线性判别分析)  
**线性判别分析(Linear Discriminant Analysis)**是一种有监督的（supervised）线性降维算法。与PCA保持数据信息不同，LDA的核心思想：往线性判别**超平面的法向量**上投影，即将数据投影到低维空间之后，使得同一类数据尽可能的紧凑，不同类的数据尽可能分散。  
## 主要思想
假设我们有两类数据，分别为红色和蓝色。如下图所示，这些数据特征是二维的，我们希望将这些数据投影到一维的一条直线，让每一种类别数据的投影点尽可能的接近，而红色和蓝色数据中心之间的距离尽可能的大。  

![](lda1.png) 
直观上可以看出，上图提供了两种投影方式，其中右图要比左图的投影效果好，因为右图的红色数据和蓝色数据各个较为集中，且类别之间的距离明显。左图则在边界处数据混杂。以上就是LDA的主要思想了，当然在实际应用中，我们的数据是多个类别的，我们的原始数据一般也是超过二维的，投影后的也一般不是直线，而是一个低维的超平面。

## 原理推导  
假设已知数据集$D=(x_1,y_1),(x_2,y_2),\dots,(x_m,y_m)$，其中样本$x_i$为任意n维向量，类别$y_i∈C_1,C_2,\dots,C_k$，定义$N_j(j∈1,2,\dots,k)$是第j类样本的个数，$X_j(j∈1,2,\dots,k)$是第j类样本的集合，$u_j(j∈1,2,\dots,k)$是第j类样本的均值向量，$\Sigma_j(j∈1,2,\dots,k)$是第j类样本的协方差矩阵。

## 二分类情况
此时k值只能为0，1两种。可得$u_j$的表达式如下：  

$$u_j=\frac{1}{N_j}\sum_{x∈X_j}x(j∈\{0,1\})$$   

$\Sigma_j$的表达式：  

$$\Sigma_j=\sum_{x∈X_j}(x-u_j)(x-u_j)^T(j∈\{0,1\})$$   

因为将其降到一维，寻找一个向量w，不同类的样本中心投影之后得：  

$$u_{wj}=wu_j$$  

不同类的方差投影之后为：  

$$
\begin{equation}
\begin{split}
\Sigma_{wj}&=\sum_{x∈X_j}(w^Tx-w^Tu_j)^2\\
&= \sum_{x∈X_j}(w^T(x-u_j))^2\\
&=\sum_{x∈X_j}w^T(x-u_j)(x-u_j)^Tw \\
&=w^T\sum_{x∈X_j}(x-u_j)(x-u_j)^Tw \\
&=w^T\Sigma_{j}w \\
\end{split}
\end{equation}
$$

为了使同一类更加聚集，不同类更加分散，我们定义如下准则公式:

$$J=\frac{||wu_0-wu_1||^2}{w^T\Sigma_0w+w^T\Sigma_1w}=\frac{w^T(u_0-u_1)(u_0-u_1)^Tw}{w^T\Sigma_0w+w^T\Sigma_1w}$$  

*注意范数没有下标，默认是2范数*  
LDA需要让不同类别的数据的类别中心之间的距离尽可能的大，也就是我们要最大化$||wu_0-wu_1||^2$，,同时我们希望同一种类别数据的投影点尽可能的接近，也就是最小化$w^T\Sigma_0w+w^T\Sigma_1w$，因此目标函数就是**最大化上式J**。  

我们一般定义类内散度矩阵为：  

$$S_w=\Sigma_0+\Sigma_1=\sum_{x∈X_0}(x-u_0)(x-u_0)^T+\sum_{x∈X_1}(x-u_1)(x-u_1)^T$$  

定义类间散度矩阵：

$$S_b=(u_0-u_1)(u_0-u_1)^T$$  

则优化目标J可以改写为： 

$$J=\frac{w^TS_bw}{w^TS_ww}$$   

由于分子和分母都是关于w的二次项，不失一般性，我们可以设：

$$w^TS_ww=1$$  

求J的最大值转换为：  

$$
\begin{equation}
\begin{aligned}
min_w \quad -w^TS_bw\\
s.t.\quad w^TS_ww=1\\
\end{aligned}
\end{equation}
$$

运用拉格朗日乘子法(具体推导和PCA类似，对拉格朗日函数求导)可得当：  

$$S_w^{-1}S_bw=\lambda w$$

时，此时函数**J**有最大值$\lambda$。

**注意**：对于二分类问题$S_bw$的方向恒平行于$u_0-u_1$,不妨令$S_bw=\lambda(u_0-u_1)$,将其带入$S_w^{-1}S_bw=\lambda w$,可以得到$w=S_w^{-1}(u_0-u_1)$,我们只需要求出原始二类样本的均值和方差就可以确定最佳的投影方向$w$。  

**了解：**根据[广义瑞利商](https://zhuanlan.zhihu.com/p/432080955)的性质，直接判断J的最大特征值和特征向量。
## 多分类情况
多分类问题推导基本和上述推导类似，不过此时类间散度矩阵为:

$$S_b=\sum_{j=1}^{k}N_j(u_j-u)(u_j-u)^T$$

其中u为所有数据点求平均值所得。  
类内散度矩阵为：  

$$S_w=\sum_{j=1}^{k}\sum_{x∈X_j}(x-u_j)(x-u_j)^T=\sum_{j=1}^{k}\Sigma_j$$   

此时设我们投影到的低维空间维度为$d$,对应的基向量为$w_1,w_2,\dots,w_d$,构成矩阵$W$。函数J的定义如下：  

$$J=\frac{W^TS_bW}{W^TS_wW}$$   

此时$J$是一个矩阵，不是一个标量，为了可以最优化我们常常使用对角线的元素来代替。此时$J$的定义如下:  

$$J=\prod_{i=1}{d}\frac{w_i^TS_bw_i}{w_i^TS_ww_i}$$    

求解方法与二分类类似，当：  

$$S_w^{-1}S_bw_i=\lambda_iw_i$$   

时，函数有最大值。所以我们取前$d$个最大的特征值对应的特征向量组成$W$。由于$S_b$是k个秩为1的矩阵相加而成，所以其秩小于等于k。又由于我们知道前k-1的$u_i$之后，最后一个$u_k$可以由前k-1个表示，因此，LDA降维算法降维之后的维度最高为k-1。  

## 举例  
对以下数据集降维：  

$$
x=
\begin{bmatrix}
1&2&0\\
3&1&0\\
-2&-2&1\\
-3&-1&1\\
\end{bmatrix}
$$  

其中每一行为一个样本，1、2列为属性，最后一列表示类别。  
**代码**：  
```python
import numpy as np

X = np.array([[1,2,0],[3,1,0],[-2,-2,1],[-3,-1,1]]) 
x = X[:,:2]
y = X[:,2]
x0 = x[y==0]
x1 = x[y==1]
def LDA(x,x1,x2):
    n1 = np.shape(x1)[0]
    n2 = np.shape(x2)[0]
    x_mean  = np.mean(x,axis=0)
    x1_mean = np.mean(x1,axis=0)
    x2_mean = np.mean(x2,axis=0)
    Sb = n1*np.dot((x1_mean-x_mean).T,(x1_mean-x_mean))+n2*np.dot((x2_mean-x_mean).T,(x2_mean-x_mean))
    Sigma1 = np.dot((x1-x1_mean).T,(x1-x1_mean))
    Sigma2 = np.dot((x2-x2_mean).T,(x2-x2_mean))
    Sw = Sigma1 + Sigma2   #特征数为n，Sb,Sw维度都是n*n
    w = np.linalg.eig(np.dot(np.linalg.inv(Sw),Sb))
    return w

eig_vals,eig_vecs = LDA(x,x0,x1)  #eig_vec每一列是对应的特征向量
# print(eig_val)
# print(eig_vec)
#特征值和特征向量配对
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# 按特征值大小进行排序
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
# 将数据降到一维。将特征向量以列向量形式拼合。
lda_matrix = np.hstack([eig_pairs[i][1].reshape(-1, 1) for i in range(1)])
# 将原始数据进行投影。
res = x.dot(lda_matrix)  #将原始数据投影
print(res)
```   
输出：
```
[[-2.22703273] 
 [-2.42784414] 
 [ 2.75276384] 
 [ 2.42784414]]
```  

## LDA算法流程  
1、计算类内散度矩阵$S_w$  
2、计算类间散度矩阵$S_B$  
3、计算矩阵$S_w^{-1}S_b$  
4、对矩阵$S_w^{-1}S_b$进行特征分解，计算最大的$d$个最大的特征值对应的特征向量组成$W$  
5、计算投影后的数据点$Y=W^TX$  


## 与PCA比较  

* PCA为无监督降维，LDA为有监督降维
* LDA降维最多降到类别数K-1的维数，PCA没有这个限制。
* PCA希望投影后的数据方差尽可能的大(最大可分性)，而LDA则希望投影后相同类别的组内方差小，而组间方差大。

　　

[参考链接：https://www.cnblogs.com/pinard/p/6244265.html](https://www.cnblogs.com/pinard/p/6244265.html)    
[多分类实例：https://www.freesion.com/article/5252242835/](https://www.freesion.com/article/5252242835/)