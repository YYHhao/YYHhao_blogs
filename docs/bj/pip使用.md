# 一、介绍
pip是一个**python包管理工具**，提供了对python包的**下载、安装、升级、卸载、查看、更新**等功能。它不仅能把我们需要的包下载下来，还会把相关依赖的包下载下来。    
因为pip是一个命令行程序，所以一般在命令行中执行相应操作。    
**Win+R,输入cmd，即打开命令行**。   
# 二、pip命令及参数    
1、在命令行窗口输入 **pip --help** ，可以查看pip命令的参数和用法     
![在这里插入图片描述](image\pip1.png)    

2、输入 **pip --version**查看pip版本    
![在这里插入图片描述](image\pip2.png)     

3、输入 **pip install**  package_name 下载指定的python包    
![在这里插入图片描述](image\pip3.png)

>pip install package_name==1.2.1 下载指定版本的包   

4、输入 **pip uninstall** package_name 卸载指定的包   
 $\qquad$用法和pip install一致   
5、输入**pip freeze**查看当前已安装的包和版本   
![在这里插入图片描述](image\pip4.png)   

6、使用 **pip list** 可以也查看已安装的库：   

![在这里插入图片描述](image\pip5.png)   

7、输入 **pip list -o** ，查看当前可升级的包   
![在这里插入图片描述](image\pip6.png)   

8、输入 **pip install -U** package_name 升级指定的包   
![在这里插入图片描述](image\pip7.png)   

9、输入**python -m pip install --upgrade pip** 升级pip   
 
10、输入 **pip show** package__name，显示指定包信息    
![在这里插入图片描述](image\pip8.png)    

11、输入 pip check package_name,验证已安装的库是否有兼容依赖问题    
![在这里插入图片描述](image\pip9.png)

12、当下载模块较大时，会超时，可以用**pip install 模块 --default-timeout=100**      



# 三、出现的问题
1、pip安装库时出现的错误，如下：   
```math
WARNING: You are using pip version 21.2.3; however, version 22.0.4 is available.
You should consider upgrading via the 'D:\python\python.exe -m pip install --upgrade pip' command.
```
解决方法，只需要在命令行输入：**python -m pip install --upgrade pip**     

![在这里插入图片描述](image\pip10.png)   

2、 安装**fitz**模块时，不能直接 pip install fitz。而是同时安装 PyMuPDF(并且是先安装 fitz 后 PyMuPDF，**顺序很重要**)  

另外注意：不能只安装PyMuPDF，当只安装PyMuPDF时，虽然可以用import fitz，但是运行fitz.open()等会出错。

