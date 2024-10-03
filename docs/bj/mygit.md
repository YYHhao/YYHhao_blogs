# 版本控制工具的作用及意义  
“代码”作为软件研发的核心产物，在整个开发周期都在递增，不断合入新需求以及解决bug的新patch，这就需要有一款系统，能够存储、追踪文件的修改历史，记录多个版本的开发和维护。于是，版本控制系统（Version Control Systems）应运而生。  
  
**版本控制工具的作用：**帮助我们记录和跟踪项目中各文件内容的修改变化。  
**记录文件修改的手工做法：**复制文件以备份，在备份的文件名中添加上日期和时间。  
**需要版本控制工具的原因：**为了提高效率，我们希望这类操作是自动进行的，这是我们需要版本控制工具的原因。    
版本控制工具（Version Control System）的分为3类：

* 直接访问式版本控制系统；
* 集中式版本控制工具，比如CVS、SVN；
* 分布式版本控制工具，比如git   

# git基本操作  
**git config**:用于配置用户，是git命令行中第一个必须使用的命令，设置了提交时要使用的作者姓名和电子邮件地址。  

**git init**:用于初始化一个本地Git仓库，Git的很多命令都需要在Git的仓库中运行，所以**git init**是使用Git的第一个命令。  

**git clone <repository URL>**:拷贝一个Git仓库到本地，让自己能够查看该项目，或者进行修改。它通过一个远程的URL访问该仓库，远程仓库的URL被称为origin。  

**git add <File name>**:用于将文件内容添加到索引（暂存区）。文件名可以有多个，用空格隔开。    

**git status**:查看仓库当前的状态，显示有变更的文件。   

**git commit  -m [message]**:将暂存区内容添加到仓库中，[message]是备注信息。  

**git push**:用于从将本地的分支版本上传到远程并合并。  

**git pull**:用于从远程获取代码并合并本地的版本,具体用法与**git push**一致。  

**git log**:查看历史提交记录。  

**git reset**:用于回退版本，可以指定退回某一次提交的版本。

**git branch**:用于创建、列出、重命名和删除分支。    

**git push <远程主机名> <本地分支名>:<远程分支名>:**:git push origin master是将本地的master分支推送到origin主机的master分支，当远程分支名与本地分支名相同，可以省略。    

**git添加所有文件**:要一次性添加仓库中所有文件，运行带有-A选项的命令**git add -A**，也可以用**git add .**。      

**添加所有修改和删除的文件**:使用**git add -u**,它允许我们只对修改和删除的文件进行分段。    

**git reset HEAD^**:回退所有内容到上一个版本。  

**git branch --list**:或**git branch**列出版本库所有可用的分支。     

**git checkout <branch name>**:在不提交的情况下从主库切换到其他任何可用的分支。   

**git branch -m master**:从任何其他分支切换到主分支。  

**git checkout -- file_name**:放弃工作区中某个文件的修改，即回退版本(将暂存区的文件恢复到工作目录)。    

**练习网站:**[https://learngitbranching.js.org/?locale=zh_CN](https://learngitbranching.js.org/?locale=zh_CN)    
**参考网站:**[https://geek-docs.com/git/git-cmds/git-push-details.html](https://www.runoob.com/git/git-basic-operations.html)    
[https://www.runoob.com/git/git-push.html](https://www.runoob.com/git/git-push.html)   

# 博客更新   
```
git add .
git commit -m 'message'
git push -u origin main
mkdocs gh-deploy --force
```
git push时出现的.ssh问题:[解决方法](https://blog.csdn.net/nightwishh/article/details/99647545)    
```
D:\python\Myblog\YYHhao_blogs>git push -u origin main
ssh: connect to host github.com port 22: Connection timed out
fatal: Could not read from remote repository.
```
git push时报错:[解决方法](https://blog.csdn.net/Skybububu/article/details/132379910)
```
fatal: unable to access 'https://github.com/YYHhao/YYHhao_blogs.git/': Failed to connect to github.com port 443 after 21159 ms: Could not connect to server
```


[mkdocs参考：https://squidfunk.github.io/mkdocs-material/publishing-your-site/#with-github-actions](https://squidfunk.github.io/mkdocs-material/publishing-your-site/#with-github-actions)
[vscode本地文件传送GitHub](https://blog.csdn.net/Libetaion/article/details/126556860)          
[GitHub文档](https://docs.github.com/en)