# super(XXX,self).__init__()的作用
在使用pytorch框架自定义网络结构时，总要有**必不可少**的第一句，super(XXX,self).\_\_init\_\_()。其实它的**作用**就是————**对继承自父类的属性进行初始化，并且用父类的初始化方法初始化继承的属性。**      

一个简单的例子：    
```python
class Person():
    def __init__(self,name,gender) -> None:
        # 初始化name，gender属性    
        self.name = name
        self.gender = gender  

    def printinfo(self):
        print(self.name,self.gender)

class Stu(Person):
    def __init__(self, name, gender,school):  #继承父类Person属性
        # 使用父类的初始化方法来初始化子类name和gender属性
        super().__init__(name, gender)
        self.school = school

    # 对父类的printinfo方法进行重写
    def printinfo(self):        
        print(self.name,self.gender,self.school)

if __name__ == '__main__':
    pe = Person('Mike','male')
    pe.printinfo()
    stu = Stu('Lucy','female','Tsinghua')
    stu.printinfo()
```
输出：
```
Mike male
Lucy female Tsinghua
```
当然，如果初始化的逻辑和父类不同，也可以自己初始化子类的属性，比如：
```python
class Person():
    def __init__(self,name,gender) -> None:
        # 初始化name，gender属性    
        self.name = name
        self.gender = gender  

    def printinfo(self):
        print(self.name,self.gender)

class Stu(Person):
    def __init__(self, name, gender,school):  
        super().__init__(name, gender)
        self.school = school
    
        #也可以对父类属性name,gender进行改写
        self.name = name.upper()   #lower()
        self.gender = gender.title()
        
    def printinfo(self):        
        print(self.name,self.gender,self.school)

if __name__ == '__main__':
    stu = Stu('Lucy','female','Tsinghua')
    stu.printinfo()
```
输出：
```
LUCY Female Tsinghua
```
然后，我们以一个简单的卷积神经网络模型lenet5为例：   
```python
class Lenet5(torch.nn.Module):
    # constructor function
    def __init__(self):
        super(Lenet5,self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5),padding=2)
        self.layer2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5),padding=0)  
        self.layer3 = torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5),padding=0)
        self.layer4 = torch.nn.Linear(120,84)  #通常用于设置网络中的全连接层
        self.layer5 = torch.nn.Linear(84,10)
```
**这里super(Lenet5, self).\_\_init\_\_()的含义：**子类Lenet5类继承父类nn.Module，super(Lenet5, self).\_\_init\_\_就是对继承自父类nn.Module的属性进行初始化。并且是用nn.Module的初始化方法来初始化继承的属性。就是**用父类nn.Module的方法初始化子类Lenet5的属性。**因为nn.Module的方法是pytorch框架中已经写好的，我们直接调用就可以，不然还要初始化各种权重和参数，太过复杂。   


**注意：**在我们创建一个类后，通常会再创建一个 \_\_init\_\_()初始化方法，当我们创建一个类的实例时，该方法就会被自动执行。       

例如,我们只是创建一个person实例，也不调用其它方法，他也会自动执行__init__()中的内容:     
```python
class Person():
    def __init__(self,name,gender) -> None:
        # 初始化name，gender属性    
        self.name = name
        self.gender = gender
        print('running')   

    def printinfo(self):
        print(self.name,self.gender)
if __name__=='__main__':
    person = Person('Mike','female')
```
输出:
```
running
```