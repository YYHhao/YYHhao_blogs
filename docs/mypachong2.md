# 一、进程、线程、协程简介
## **进程(Process)：**
进程是指计算机中已运行的程序，是系统进行资源分配和调度的基本单位，是操作系统结构的基础。

1.进程就是一个程序的执行实例，也就是正在执行的程序。在OS的眼里，进程就是一个担当分配系统资源CPU时间、内存的实体。
![在这里插入图片描述](https://img-blog.csdnimg.cn/c7b26968dd8c4707b4b4e21ef9b7f4b5.png)

2.进程控制的主要功能是对系统中的所有进程实施有效的管理，它具有创建新进程、撤销已有进程、实现进程状态之间的转换等功能。
![在这里插入图片描述](https://img-blog.csdnimg.cn/690cfdcd681c4aa88d101517fc6bf997.png)


3.进程在运行中不断地改变其运行状态。一个进程在运行期间，不断地从一种状态转换到另一种状态，它可以多次处于就绪状态和执行状态，也可以多次处于阻塞状态。
![在这里插入图片描述](https://img-blog.csdnimg.cn/0e3ce9afb6d942198d09dc1aa6647229.png)
>简而言之，进程就好比你的手机同时用着QQ和微信，这就是两个进程。
## **线程(Thread):**
线程是操作系统能够进行运算调度的最小单位。它被包含在进程之中，是进程中的实际运作单位。

一条线程指的是进程中一个单一顺序的控制流，一个进程中可以并发多个线程，每条线程并行执行不同的任务。在Unix System V及SunOS中也被称为轻量进程（lightweight processes），但轻量进程更多指内核线程（kernel thread），而把用户线程（user thread）称为线程。
![在这里插入图片描述](https://img-blog.csdnimg.cn/64fb29fdb73f4109b30629c045beeb51.png)

>简而言之，线程就是你用手机登着QQ，可以一边和别人视频一边发信息，这就是在一个进程中的两个不同线程。

## **协程(Corouties):**
协程是一种比线程更加轻量级的一种函数。

操作系统在线程等待IO的时候，会阻塞当前线程，切换到其它线程，这样在当前线程等待IO的过程中，其它线程可以继续执行。当系统线程较少的时候没有什么问题，但是当线程数量非常多的时候，却产生了问题。**一是系统线程会占用非常多的内存空间；二是过多的线程切换会占用大量的系统时间。**
协程的出现有效解决上述两个问题，协程运行在线程之上，当一个协程执行完成后，可以选择主动让出，让另一个协程运行在当前线程之上。**协程并没有增加线程数量，只是在线程的基础之上通过分时复用的方式运行多个协程，** 而且协程的切换在用户态完成，切换的代价比线程从用户态到内核态的代价小很多。
> 简而言之，就好比你去足疗正自己洗脚，技师要走，而你让他帮忙按摩一下肩膀，不让他走,一会洗好了再按脚。
 **注意：协程只有和异步IO结合起来，才能发挥最大的威力。**

1、程序中产生的阻塞状态，以下方代码为例
```python
import time

def func():
    print('a')
    time.sleep(2)  #让当前线程处于阻塞状态。CUP不为我工作
    print('b')
    
if __name__ == '__main__':
    func()

#input()程序也是处于阻塞状态
#requests.get()在网络请求返回数据之前，程序也是处于阻塞状态的
#一般情况下，当程序处于IO操作的时候，线程都会处于阻塞状态
#协程：当程序遇见了IO操作的时候，可以选择性的切换到其它任务上
```
2、协程用到asyncio模块，如下：

```python
import asyncio
import time
async def func():
    print('Hello,world')

if __name__ == '__main__':
    g = func() #此时的函数是异步协程函数，此时函数执行得到的是一个协程对象
    # print(g)
    asyncio.run(g)    #协程程序运行需要asyncio模块支持
```
输出：

```cpp
<coroutine object func at 0x0000020AAB9EEF80>
Hello,world
```
3、一个采用协程的实例
```python
import asyncio
import time
async def func1():
    print('你好,嗡嗡嗡')
    await asyncio.sleep(3) 
    print('你好,水水水')

async def func2():
    print('你好,顶顶顶')
    await asyncio.sleep(2)
    print('你好,哈哈哈')

async def func3():
    print('你好,啊啊啊')
    await asyncio.sleep(4)
    print('你好,嘻嘻嘻')

async def main():
    #第一种写法
    # f1 = func1()
    # await f1   #一般await挂起操作放在协程对象前面
    #第二种写法(推荐)
    tasks = [
        asyncio.create_task(func1()),    #py3.8以后先把协程对象包装成task
        asyncio.create_task(func2()),
        asyncio.create_task(func3())
    ]
    await asyncio.wait(tasks)   #多个任务一般第一是asyncio.wait()
if __name__ == '__main__':  
    t1 = time.time()
    asyncio.run(main())
    t2 = time.time()
    print(t2-t1)

#协程耗时应该和最大时间对应
```
输出：

```cpp
你好,嗡嗡嗡
你好,顶顶顶
你好,啊啊啊
你好,哈哈哈
你好,水水水
你好,嘻嘻嘻
4.016979932785034
```

# 二、多进程、多线程
## 多进程：
同一时刻并行的处理多个任务，即为多进程。
比如，你一边喝茶、看书还听着音乐。真正的并行多任务只能在多核的CPU上实现，由于任务数量是远远多于CPU的核数，所以操作系统会自动将多任务短时间轮流切换执行，给我们的感觉就像同时在执行一样。

代码实例
```cpp
from multiprocessing import Process
def func():
    for i in range(100):
        print('子进程',i)

if __name__ == '__main__':
    p = Process(target=func)
    p.start()
    for i in range(100):
        print("主进程",i)
```
输出：

```cpp
主进程 0
主进程 1
主进程 2
...
子进程 97
子进程 98
子进程 99
```
## 多线程：
多线程就是把操作系统中的这种并发执行机制原理运用在一个程序中，把一个程序划分为若干个子任务，多个子任务并发执行，每一个任务就是一个线程。因为CPU在执行程序时每个时间刻度上只会存在一个线程，因此多线程实际上提高了进程的使用率从而**提高了CPU的使用率**。**但线程过多时，操作系统在他们直接来回切换，影响性能。**

1、一般的多线程
```python
from threading import Thread
def func1():
    for i in range(100):
        print('func1',i)

def func2():
    for i in range(100):
        print('func2',i)

if __name__ == '__main__':
    t1 = Thread(target=func1)  #创建线程，并给线程安排任务，（子线程）
    t1.start()  #多线程状态为可以开始工作状态，具体执行时间由CPU决定
    t2 = Thread(target=func2)
    t2.start()
    for i in range(100):
        print('main',i) 
```
输出：

```cpp
...
func1 88
func1 89
func1 90
...
main func1
83 func2 69
91
func2
...
```
2、创建多线程类

```python
class MyThread(Thread):
    def run(self):     #固定的
        for i in range(100):
            print('子线程',i)

if __name__ == '__main__':
    t = MyThread()
    # t.run()是方法的调用，不行，是单线程
    t.start() #开启线程
    for i in range(100):
        print("主线程",i)
```

3、给线程传入参数

```cpp
def func(name):
    for i in range(100):
        print(name,i)

if __name__ == '__main__':
    t1 = Thread(target=func,args=('周杰伦',))  
    t1.start()
    t2 = Thread(target=func,args=('张杰',))
    t2.start()
    for i in range(100):
        print('main',i) 
```

#  三、线程池、进程池
## **池的概念：**
池是保证计算机能够正常运行的前提下，能最大限度开采的资源的量，他降低的程序的运行效率，但是保证了计算机硬件的安全，从而让你写的程序可以正常运行。池就是一个量的单位，代表了你所能开采的资源的度，超过这个度，身体就跨了，首先要用，其次要有度。
**采用线程池和进程池：保证硬件能够正常工作的情况下，最大限度的利用资源。**
## 线程池：
线程池是一种多线程处理形式，处理过程中将任务添加到队列，然后在创建线程后自动启动这些任务。
**采用ThreadPoolExecutor（3），括号内是创建的线程数，不填的话，会默认开设当前计算机cpu核数的5倍的线程。**

```python
#线程池：一次性开辟一些线程。我们用户直接给线程池子提交任务。线程任务的调度交给线程池完成。
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor

def fn(name):
    for i in range(1000):
        print(name,i)

if __name__ == '__main__':
    #创建线程池
    with ThreadPoolExecutor(50) as t:
        for i in range(100):
            t.submit(fn,name=f"线程{i}")
    #等待线程池中任务全部执行完毕，才继续执行（守护）
    print('123')
```

## 进程池：
众所周知，Python 的多线程是假的，不过好在开发者老大还是给我们留了一个活路，也就是进程池。这个方法的优点在于进程的并发细节完全不用我们操心，我们只需要把并发的任务仍到进程池里就好了。
**进程池大体上跟线程池类似，ProcessPoolExecutor(3)，括号内不填的话，会默认创建与“cpu核数”相同数量的进程。** 同样的，进程池中的进程是固定工，不会重复创建和销毁。

进程池和线程池在使用形式上是一样的，唯一不同的是：在Windows环境下，进程池要放在main方法里面，否则会报错。

```python 
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor

def fn(name):
    for i in range(10):
        print(name,i)

if __name__ == '__main__':
    #创建进程池
    with ProcessPoolExecutor(50) as t:
        for i in range(100):
            t.submit(fn,name=f"进程{i}")
```
输出：

```cpp
进程0 0
进程0 1
进程0 2
进程0 3
进程0 4
进程0 5
进程0 6
进程0 7
进程0 8
进程0 9
...
```