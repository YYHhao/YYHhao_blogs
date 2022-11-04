# pandas简介     
Pandas (Python Data Analysis Library) 是基于NumPy 的一种工具，该工具是为了解决数据分析任务而创建的。     

numpy更加适用于科学计算领域；而pandas最初据说是设计用于金融领域，因此pandas可能更加适用于各类实际应用场景的处理。另一方面，**numpy适用于处理“干净”的数据，及规范、无缺失的数据**，而**pandas更加擅长数据清洗(data munging)**，这为后一步数据处理扫清障碍。    

Excel可以处理少量的数据，当数据量较大时许多操作无法进行，这就需要利用pandas模块进行数据处理。   
# pandas中的python数据结构    

* Series:一堆数组。与python标准数据结构list列表类似。   
* DataFrame:二维的表格型数据结构。可以理解为Series容器。   
* Panel:三维数组，可以理解为DataFrame容器。  

# 常见的pandas读取数据类型   

* pandas.read_csv()  
* pandas.read_excel()  
* pandas.read_sql()   
* pandas.read_html()  
* pandas.read_json()  

# Series的创建与索引   
1、      
```python
import pandas as pd

#通过数组创建Series
s1 = pd.Series([1,2,3,'Tom',True])
print(s1)

#创建指定索引列的Series
s2 = pd.Series(['tom','jack','dane'],index=['001','002','003'])
print(s2)

#使用字典创建Series
s3 = pd.Series({'tom':'001','jack':'002'})
print(s3)

#获取s1的索引列
print(s1.index)

#获取s1的值
print(s1.values)

# 获取多个索引的值
print(s1[[0,1]])
```
2、
```python
s = pd.Series([1,2,3,2,4],index=['a','b','c','d','e'])
print(s)
print('\n')
print(s.max())  #最大
print(s.min())  #最小
print(s.median()) #中位数
print(s.sum())  #求和

s.drop('d')#删除索引为d的值
s['a'] = 0 #修改索引为a的值
print(s)

s1 = pd.Series([5,4,3,2,1],index=list(map(lambda x:x.upper(),['a','b','c','d','e'])))
print('\n')
print(s1)
s2 = s.append(s1)  #将s1合并到s后
print(s2)
```
```
a    1      
b    2      
c    3      
d    2      
e    4      
dtype: int64

4  
1  
2.0
12

a    0
b    2
c    3
d    2
e    4
dtype: int64

A    5
B    4
C    3
D    2
E    1
dtype: int64

a    0
b    2
c    3
d    2
e    4
A    5
B    4
C    3
D    2
E    1
dtype: int64
```


# dataframe的基本操作    

## 创建dataframe   
1、 一般使用pandas直接读取文件，就是dataframe类型    
```python
pd.read_csv()
pd.read_excel()
pd.read_sql()
```  
2、通过字典来创建    
```python
import pandas as pd
data = {'id':[1,2,3],'name':['tom','jack','luck']}
df = pd.DataFrame(data)
print(df)
```
结果:
```
   id  name
0   1   tom
1   2  jack
2   3  luc
```
## dataframe基本操作   
```python
#获取数据类型
print(df.dtypes)

#获取columns
print(df.columns)

#获取索引列
print(df.index)
```
输出：
```
id       int64
name    object
dtype: object

Index(['id', 'name'], dtype='object')

RangeIndex(start=0, stop=3, step=1)
```


```python
#获取某一列
print(df['id'])
print(type(df['id']))  #series

#获取多列数据,按索引
print(df[['name','id']])
print(type(df[['name','id']]))  #DataFrame

#获取前n条数据
print(df.head(1))

#获取后n条数据
print(df.tail(2))

#获取某一行数据
df.loc[1]

#获取多行数据
df.loc[1:2]
```