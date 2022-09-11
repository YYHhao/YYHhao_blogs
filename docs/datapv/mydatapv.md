**在默认情况下，matplotlib无法在图表中使用中文**  
 一、直接在fontproperties属性后加字体类型,但字体类型有限。  
```python
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
x = np.linspace(0,10,100)
plt.plot(x,np.sin(x),label='sin(x)')
plt.plot(x,np.cos(x),label='cos(x)函数')
plt.xlabel('x刻度',fontproperties='stsong')
plt.ylabel('y刻度',fontproperties='SimHei')
plt.legend(prop='SimHei')       #给图像加个图例
plt.show()
```  
![在这里插入图片描述](https://img-blog.csdnimg.cn/96812e0214b4463b9783bffb87e7fb37.png#pic_center)  

二、先找到自己电脑上字体路径，然后调用font_manager模块中的FontProperties类，存储和操作字体属性，更加全面。  

```python
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
font_path = 'C:\Windows\Fonts\simfang.ttf'
my_font = mpl.font_manager.FontProperties(fname=font_path)
x = np.linspace(0,10,100)
plt.plot(x,np.sin(x))
plt.title('sin函数',fontproperties=my_font)
plt.xlabel('x刻度',fontproperties=my_font)
plt.ylabel('y刻度',fontproperties=my_font)
plt.show() 
```   
![在这里插入图片描述](https://img-blog.csdnimg.cn/f9108e0e34a943a194c01ebf4d08fe10.png#pic_center)    

[font_manager模块FontProperties类的使用：https://blog.csdn.net/mighty13/article/details/116243372](https://blog.csdn.net/mighty13/article/details/116243372)