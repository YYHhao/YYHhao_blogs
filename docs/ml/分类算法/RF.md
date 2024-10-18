# 随机森林
Random Forest(随机森林)是一种基于树模型的优化版本，一棵树的生成肯定还是不如多棵树，因此就有了随机森林，解决决策树泛化能力弱的特点。其实就是利用集成学习方法生成多个决策树分类器，让所有决策树进行预测，选出结果中出现最多的预测结果。

# 常用的随机过程
## Bootstrapping
Bootstrapping算法，指的就是利用有限的样本资料经由多次有放回的重复抽样。如：在原样本中有放回的抽样，抽取n次。每抽一次形成一个新的样本，重复操作，形成很多新样本。

1、有放回采样

2、强调偏差

3、串行执行，速度较慢

4、可以提升泛化性能
## Bagging
Bagging就是bootstrap aggregation：从样本集（假设样本集N个数据点）中随机重采样选出N个样本（有放回的采样，样本数据点个数仍然不变为N），在所有样本上，对这n个样本建立分类器（ID3\C4.5\CART\SVM\LOGIST
IC），重复以上两步m次，获得m个分类器，最后根据这m个分类器的投票结果，对于分类任务，通常采用多数投票或平均概率决定最终类别；对于回归任务，采用平均值作为集成模型的预测结果。


1、均匀采样

2、强调方差

3、并行生成，速度快

4、可以提升泛化性能

bagging举个例子：

假设有1000个样本，如果按照以前的思维，是直接把这1000个样本拿来训练，但现在不一样，先抽取800个样本来进行训练，假如噪声点是这800个样本以外的样本点，就很有效的避开了。重复以上操作，提高模型输出的平均值。

# 随机森林影响因素
**1、随机森林分类效果的影响因素**   
* 森林中任意两棵树的相关性：相关性越大，错误率越大；     
* 森林中每棵树的分类能力：每棵树的分类能力越强，整个森林的错误率越低。   

减小特征选择个数m，树的相关性和分类能力也会相应的降低；增大m，两者也会随之增大。所以关键问题是如何选择最优的m（或者是范围），这也是随机森林唯一的一个参数。

# 随机森林优缺点
**1、优点：**
RF简单，容易实现，计算开销小，性能强大。它的扰动不仅来自于样本扰动，还来自于属性扰动，这使得它的泛化性能进一步上升。

**2、缺点：**
它在训练和预测时都比较慢，而且如果需要区分的类别很多时，随机森林的表现并不会很好。

# 算法实现

sklearn.ensemble.RandomForestClassifier(n_estimators = 10,criterion = 'gini',max_depth = None,bootstrap = True,random_state = None,min_samples_split = 2)

* 随机森林分类器
* n_estimators：整数类型，指森林里的树木数量，默认为10。通常为120,200,300,500,800,1200。
* criterion：字符串类型，默认基尼系数'gini'，你也可以换成熵。
* max_depth：整数类型，默认None，用于指定树的最大深度。通常为5,8,15,25,30。
* max_features ：字符串类型，每个决策树的最大特征数量，有'auto','sqrt','log2','None'四种选择。
* bootstrap：布尔类型，默认开启，指是否在构建森林时使用返回抽样
* min_samples_split：结点划分最少样本数
* min_samples_leaf：叶子结点的最小样本数。
* 其中n_estimator,max_depth,min_samples_split,min_samples_leaf可以使用超参数网格搜索进行调参。

```
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def decision_iris():
    """随机森林算法"""
    # 1 获取数据集
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=4)

    # 2 预估器实例化
    estimator = RandomForestClassifier(criterion="entropy")

    # 3 指定超参数集合
    param_dict = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}

    # 4 加入超参数网格搜索和交叉验证
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)

    # 5 训练数据集
    estimator.fit(x_train, y_train)

    # 6 模型评估
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接对比真实值和预测值：\n", y_test == y_predict)

    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)


decision_iris()
```
输出：
```
y_predict:
 [2 0 2 2 2 1 2 0 0 2 0 0 0 1 2 0 1 0 0 2 0 2 1 0 0 0 0 0 0 2 1 0 2 0 1 2 2
 1]
直接对比真实值和预测值：
 [ True  True  True  True  True  True False  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True  True
  True  True]
准确率为：
 0.9736842105263158

```


[浅显易懂的机器学习（八）—— 随机森林分类](https://developer.aliyun.com/article/1051613)   
[网格搜索调超参数](https://blog.csdn.net/weixin_41988628/article/details/83098130)