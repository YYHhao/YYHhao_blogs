## Mysql数据库常用术语  
**数据库:** 数据库是一些关联表的集合。   
**数据表:** 表是数据的矩阵。在一个数据库中的表看起来像一个简单的电子表格。   
**列:** 一列(数据元素) 包含了相同类型的数据, 例如邮政编码的数据。   
**行:** 一行（元组，或记录）是一组相关的数据，例如一条用户订阅的数据。  
**冗余:** 存储两倍数据，冗余降低了性能，但提高了数据的安全性。   
**主键:** 主键是唯一的。一个数据表中只能包含一个主键。你可以使用主键来查询数据。    
**外键:** 外键用于关联两个表。  
**复合键:** 复合键（组合键）将多个列作为一个索引键，一般用于复合索引。  
**索引:** 使用索引可快速访问数据库表中的特定信息。索引是对数据库表中一列或多列的值进行排序的一种结构。类似于书籍的目录。   
**参照完整性:** 参照的完整性要求关系中不允许引用不存在的实体。与实体完整性是关系模型必须满足的完整性约束条件，目的是保证数据的一致性。   
**表头(header):** 每一列的名称。       
**列(col):** 具有相同数据类型的数据的集合。   
**行(row):** 每一行用来描述某条记录的具体信息。   
**值(value):** 行的具体信息, 每个值必须与该列的数据类型相同。   
**键(key):** 键的值在当前列中具有唯一性。   

## 命令行操作Mysql
**1、登陆Mysql**
```
mysql -h localhost -u root -P 3306 -p 
#输入密码 Enter password:******
```
-h mysql连接地址

-u mysql登录用户名

-P mysql连接端口(默认为 3306)

-p mysql登录密码

**2、查看数据库**
```
show databases;
```
**3、创建数据库**
```
CREATE DATABASE [IF NOT EXISTS] database_name
  [CHARACTER SET charset_name]
  [COLLATE collation_name];
```

**4、删除数据库**
```
DROP DATABASE db_name;　　#删除
```

**5、查询数据**
```
SELECT * FROM tb1 WHERE name='李四';
```

## Mysql Workbench中创建字段字符含义     
* **PK(primary key):** 主键    
* **NN(not null):** 非空    
* **UQ(unique):** 唯一索引   
* **BIN(binary):** 二进制数据(比text更大)   
* **UN(unsigned):** 无符号（非负数）   
* **ZF(zero fill):** 填充0 例如字段内容是1 int(4), 则内容显示为0001   
* **AI(auto increment):** 自增     
* **G(generated column):** 生成列

## Mysql删除数据行    

**1.DELETE语句**

MySQL中的DELETE语句用于删除一个或多个数据行。语法如下：

DELETE FROM table_name

WHERE condition;

其中，table_name是要删除数据行的表名，condition是删除数据行的条件。例如，如果要删除名为“students”的表中姓氏为“张”的学生记录，可以使用以下命令：

DELETE FROM students WHERE last_name = ‘张’;

这条命令将删除表中所有姓氏为“张”的学生记录。

**2. TRUNCATE TABLE语句**

TRUNCATE TABLE语句可以删除一个表中的所有数据行，并且重置自增长的主键值。语法如下：

TRUNCATE TABLE table_name;

例如，如果要删除名为“students”的表中的所有数据行，可以使用以下命令：

TRUNCATE TABLE students;

这条命令将删除表中所有数据行。

**3. DROP TABLE语句**

DROP TABLE语句用于删除一个表及其所有数据行。语法如下：

DROP TABLE table_name;

例如，如果要删除名为“students”的表及其所有数据行，可以使用以下命令：

DROP TABLE students;

这条命令将删除表及其所有数据行。

需要注意的是，DROP TABLE语句将永久删除表及其所有数据，因此在使用前需要谨慎考虑。

总结：在MySQL中，通过DELETE语句可以删除单个或多个数据行，TRUNCATE TABLE语句可以删除一个表中的所有数据行，并重置自增长的主键值，而DROP TABLE语句可以删除表及其所有数据行。在使用这些语句时，需要谨慎考虑，避免不必要的数据丢失。

## pymysql中executemany批量插入数据

1、在写sql语句时，不管字段为什么类型，占位符统一使用%s,且不能加上引号。例如

```
sql="insert into tablename (id,name) values (%s,%s)
```
2、添加的数据的格式必须为 **list[tuple(),tuple(),tuple()]** 或者**tuple(tuple(),tuple(),tuple())** 例如
```
values=[(1,"zhangsan"),(2,"lisi")]
values=((1,"zhangsan"),(2,"lisi"))
```

3、豆瓣Top250将电影名称、评分、评价人数保存到mysql 

```
#保存到mysql
mysql_local = {
    'host':'localhost',
    'port':3306,
    'user':'root',
    'password':'082008271005yyh',
    'db':'movietop250',
}
#建立数据库连接
conn = pymysql.connect(**mysql_local)
cursor = conn.cursor()

def save_mysql(datalist):
    film_names = []
    scores = []
    numbers = []
    for i in range(250):
        film_names.append(datalist[i][2])
        scores.append(datalist[i][4])
        numbers.append(datalist[i][5])
    datalists = list(zip(film_names,scores,numbers))
    sql = "insert into movietop250(film_name,score,number) values(%s,%s,%s)"
    cursor.executemany(sql,datalists)   #执行数据库命令
    conn.commit()  #提交    
```
