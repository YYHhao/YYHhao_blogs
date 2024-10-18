# 梯度提升决策树(CBDT)      
GBDT（Gradient Boosting Decision Tree），全名叫梯度提升决策树，是一种迭代的决策树算法，又叫 MART（Multiple Additive Regression Tree），它通过构造一组弱的学习器（树），并把多颗决策树的结果累加起来作为最终的预测输出。该算法将决策树与集成思想进行了有效的结合。

GBDT中的树是回归树（不是分类树），GBDT用来做回归预测，调整后也可以用于分类。

# Boosting核心思想
Boosting方法训练基分类器时采用串行的方式，各个基分类器之间有依赖。它的基本思路是将基分类器层层叠加，每一层在训练的时候，对前一层基分类器分错的样本，给予更高的权重。测试时，根据各层分类器的结果的加权得到最终结果。

Bagging 与 Boosting 的串行训练方式不同，Bagging 方法在训练过程中，各基分类器之间无强依赖，可以进行并行训练。

# GBDT优缺点
**优点**   
* 预测阶段，因为每棵树的结构都已确定，计算速度快。
* 适用稠密数据，泛化能力和表达能力都不错，数据科学竞赛榜首常见模型。
* 可解释性不错，鲁棒性亦可，能够自动发现特征间的高阶关系。    

**缺点**
* GBDT 在高维稀疏的数据集上，效率较差，且效果表现不如 SVM 或神经网络。
* 适合数值型特征，在 NLP 或文本特征上表现弱。
* 训练过程无法并行，工程加速只能体现在单颗树构建过程中。
# 随机森林 vs GBDT
**相同点**
* 都是集成模型，由多棵树组构成，最终的结果都是由多棵树一起决定。
* RF 和 GBDT 在使用 CART 树时，可以是分类树或者回归树。

**不同点**
* 训练过程中，随机森林的树可以并行生成，而 GBDT 只能串行生成。
* 随机森林的结果是多数表决表决的，而 GBDT 则是多棵树累加之。
* 随机森林对异常值不敏感，而 GBDT 对异常值比较敏感。
* 随机森林降低模型的方差，而 GBDT 是降低模型的偏差。
# 算法原理
* 所有弱分类器的结果相加等于预测值。
* 每次都以当前预测为基准，下一个弱分类器去拟合误差函数对预测值的残差（预测值与真实值之间的误差）。
* GBDT的弱分类器使用的是树模型(cart)。
# 代码实现
sklearn.ensemble.GradientBoostingClassifier(*, loss='log_loss', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
* loss：损失函数。有log_loss和exponential两种。'log_loss' 是指对数损失，与 Logistic 回归中使用的相同，适用于于具有概率输出的分类任务。exponential是指数损失，后者相当于AdaBoost。   

* n_estimators:最大弱学习器个数，默认是100，调参时要注意过拟合或欠拟合，一般和learning_rate一起考虑。
* learning_rate:步长，即每个弱学习器的权重缩减系数，默认为0.1，取值范围0-1，当取值为1时，相当于权重不缩减。较小的learning_rate相当于更多的迭代次数。
* subsample:子采样，默认为1，取值范围(0,1]，当取值为1时，相当于没有采样。小于1时，即进行采样，按比例采样得到的样本去构建弱学习器。这样做可以防止过拟合，但是值不能太低，会造成高方差。
* init：初始化弱学习器。不使用的话就是第一轮迭代构建的弱学习器.如果没有先验的话就可以不用管

由于GBDT使用CART回归决策树。以下参数用于调优弱学习器，主要都是为了防止过拟合
* max_feature：树分裂时考虑的最大特征数，默认为None，也就是考虑所有特征。可以取值有：log2,auto,sqrt
* max_depth：CART最大深度，默认为None
* min_sample_split：划分节点时需要保留的样本数。当某节点的样本数小于某个值时，就当做叶子节点，不允许再分裂。默认是2
* min_sample_leaf：叶子节点最少样本数。如果某个叶子节点数量少于某个值，会同它的兄弟节点一起被剪枝。默认是1
* min_weight_fraction_leaf：叶子节点最小的样本权重和。如果小于某个值，会同它的兄弟节点一起被剪枝。一般用于权重变化的样本。默认是0
* min_leaf_nodes：最大叶子节点数

```

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

gbdt = GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, n_estimators=5, subsample=1
                                  , min_samples_split=2, min_samples_leaf=1, max_depth=3
                                  , init=None, random_state=None, max_features=None
                                  , verbose=0, max_leaf_nodes=None, warm_start=False
                                  )

train_feat = np.array([[1, 5, 20],
                       [2, 7, 30],
                       [3, 21, 70],
                       [4, 30, 60],
                       ])
train_label = np.array([[0], [0], [1], [1]]).ravel()

test_feat = np.array([[5, 25, 65]])
test_label = np.array([[1]])
print(train_feat.shape, train_label.shape, test_feat.shape, test_label.shape)

gbdt.fit(train_feat, train_label)
pred = gbdt.predict(test_feat)

total_err = 0
for i in range(pred.shape[0]):
    print(pred[i], test_label[i])
    err = (pred[i] - test_label[i]) / test_label[i]
    total_err += err * err
print("总误差：",total_err / pred.shape[0])
```
输出：
```
(4, 3) (4,) (1, 3) (1, 1)
1 [1]
总误差： [0.]
```
[深入解析GBDT二分类算法](https://blog.csdn.net/weixin_39910711/article/details/80100100)    
[梯度提升决策树](https://blog.csdn.net/ShowMeAI/article/details/123402422)