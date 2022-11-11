# numpy.nonzero()
nonzero()返回矩阵或数组元素中不是False和0的索引，返回的第一个array描述行，第二个array描述列     

```python
import numpy as np 

matrix = np.mat([[1,0,2],[0,3,0],[0,0,0],[1,5,1]])
print(matrix)
#将矩阵转化为numpy中的数组
print(matrix.A)   
#提取第一列
print(matrix[:,0])   
#返回第一列非零元素索引
print(np.nonzero(matrix[:,0]==1))

# 提取出第一个数值为1的全部行
print(matrix[np.nonzero(matrix[:,0]==1)[0]])
```
```
[[1 0 2] 
 [0 3 0] 
 [0 0 0] 
 [1 5 1]]

[[1 0 2] 
 [0 3 0] 
 [0 0 0] 
 [1 5 1]]

[[1]     
 [0]
 [0]
 [1]]

(array([0, 3], dtype=int64), array([0, 0], dtype=int64))

[[1 0 2]
 [1 5 1]]
```