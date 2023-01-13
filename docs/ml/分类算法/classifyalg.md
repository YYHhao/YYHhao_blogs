# 分类算法  
分类是一个**有监督**的学习过程，目标数据库中有哪些类别是已知的，分类过程需要做的就是把每一条记录归到对应的类别之中。由于必须事先知道各个类别的信息，并且所有待分类的数据条目都默认有对应的类别，因此分类算法也有其局限性，当上述条件无法满足时，我们就需要尝试聚类分析。
# 分类和聚类的区别
聚类和分类是两种不同的分析。  

**分类的目的是为了确定一个点的类别，具体有哪些类别是已知的**，常用的算法是 KNN (k-nearest neighbors algorithm)，是一种有监督学习。**聚类的目的是将一系列点分成若干类，事先是没有类别的**，常用的算法是 K-Means 算法，是一种无监督学习。

两者也有共同点，那就是它们都包含这样一个过程：对于想要分析的目标点，都会在数据集中寻找离它最近的点，即二者都用到了 NN (Nears Neighbor) 算法。

# 常用的分类算法

* K近邻(KNN)
* 线性回归(Linear Regression)
* 逻辑回归(Logistic Regression)
* 人工神经网络( Artificial Neural Network,简称ANN)
* 支持向量机(SVM)
* 条件随机场(CRF)
* 朴素贝叶斯分类(Naive Bayes Classifier，简称NBC)
* 决策树(包括ID3算法、C4.5算法、C5.0算法)
* 随机森林分类器(Random Forest Classifier)
* 分类回归树(Classification and Regression Tree，简称CART)
* 梯度提升决策树((Gradient Boosting Decision Tree，简称CBDT)
