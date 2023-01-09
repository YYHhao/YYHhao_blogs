# numpy.argmax(array,axis=None)
numpy.argmax() 函数是沿给定轴返回最大元素的索引，同理 numpy.argmin()函数是沿给定轴返回最小元素的索引。

>axis：默认情况下，不指定axis时，会把array铺平，然后返回其中最大值的索引；
axis=0时，返回每列中最大值的索引；
axis=1时，返回每行中最大值的索引；

```python
import numpy as np
a = np.array([[2,14,3,6],[5,2,9,14]])
print('数组平铺时最大值的索引:',np.argmax(a))
print('数组每行中最大值的索引:',np.argmax(a,axis=1))
print('数组每列中最大值的索引:',np.argmax(a,axis=0))
```
输出：  
```
数组平铺时最大值的索引: 1
数组每行中最大值的索引: [1 3]
数组每列中最大值的索引: [1 0 1 1]
```
**注意：**当数组中最大值个数不唯一时，只返回第一个最大值的索引。