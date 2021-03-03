## 第 13 课 Pandas (2)

### 课程纲要

#### 索引对象

- 不可变的对象，便于赋值

  ```python
  obj = pd.Series(range(3), index=['a', 'b', 'c'])
  index = obj.index
  # index[1] = 'd'
  index = pd.Index(np.arange(3))
  obj2 = pd.Series([1.5, -2.5, 0], index=index)
  print(obj2.index is index)
  #	True
  ```

- 重新索引 Series（填充默认值、修改默认值填充方法）

  ```python
  obj = pd.Series([4, 7, 0.3, 3], index=['a', 'b', 'c', 'd'])
  print(obj)
  # a    4.0
  # b    7.0
  # c    0.3
  # d    3.0
  # dtype: float64
  obj = obj.reindex(['a', 'b', 'c', 'd', 'e'])
  print(obj)
  # a    4.0
  # b    7.0
  # c    0.3
  # d    3.0
  # e    NaN
  # dtype: float64
  
  '''填充默认值'''
  obj = obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)
  # a    4.0
  # b    7.0
  # c    0.3
  # d    3.0
  # e    0.0
  # dtype: float64
  
  '''修改默认值填充方法 ffill  bfill'''
  obj = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
  print(obj.reindex(range(6), method='ffill'))
  # 0      blue
  # 1      blue
  # 2    purple
  # 3    purple
  # 4    yellow
  # 5    yellow
  # dtype: object
  ```

- 重新索引 DataFrame（索引 index, coloums）

  ```python
  src = {
      'Hubei': ['Shiyan', 'Xiaogan', 'Hankou'],
      'Hunan': ['Yueyang', 'Changsha', 'Xiangtan'],
      'Qinghai': ['Haibei', 'Haidong', 'Haixi']
  }
  df = pd.DataFrame(src)
  df.index = pd.Series(['a', 'b', 'c'])
  print(df)
  #      Hubei     Hunan  Qinghai
  # a   Shiyan   Yueyang   Haibei
  # b  Xiaogan  Changsha  Haidong
  # c   Hankou  Xiangtan    Haixi
  
  df = df.reindex(['b', 'a', 'c'])
  #      Hubei     Hunan  Qinghai
  # b  Xiaogan  Changsha  Haidong
  # a   Shiyan   Yueyang   Haibei
  # c   Hankou  Xiangtan    Haixi
  
  states = ['Hubei', 'Hunan', 'Qinghai', 'Shanghai']
  df = df.reindex(columns=states)
  print(df)
  #      Hubei     Hunan  Qinghai  Shanghai
  # a   Shiyan   Yueyang   Haibei       NaN
  # b  Xiaogan  Changsha  Haidong       NaN
  # c   Hankou  Xiangtan    Haixi       NaN
  ```


### 查看数据

- 切片

  ```python
  #      Hubei     Hunan  Qinghai
  # a   Shiyan   Yueyang   Haibei
  # b  Xiaogan  Changsha  Haidong
  # c   Hankou  Xiangtan    Haixi
  
  #	查找行
  print(df[0:2])
  print(df['a': 'b'])
  
  #      Hubei     Hunan  Qinghai
  # a   Shiyan   Yueyang   Haibei
  # b  Xiaogan  Changsha  Haidong
  
  #	查找列
  print(df['Hubei'])
  # a     Shiyan
  # b    Xiaogan
  # c     Hankou
  # Name: Hubei, dtype: object
  ```

- 标签（按index）

  ```python
  #      Hubei     Hunan  Qinghai
  # a   Shiyan   Yueyang   Haibei
  # b  Xiaogan  Changsha  Haidong
  # c   Hankou  Xiangtan    Haixi
  
  print(df.loc['a'])
  # Hubei       Shiyan
  # Hunan      Yueyang
  # Qinghai     Haibei
  # Name: a, dtype: object
  
  print(df.loc['a': 'b', "Hunan":'Qinghai'])
  #       Hunan  Qinghai
  # a   Yueyang   Haibei
  # b  Changsha  Haidong
  ```

- 位置

  ```python
  #      Hubei     Hunan  Qinghai
  # a   Shiyan   Yueyang   Haibei
  # b  Xiaogan  Changsha  Haidong
  # c   Hankou  Xiangtan    Haixi
  
  print(df.iloc[1])
  # Hubei       Xiaogan
  # Hunan      Changsha
  # Qinghai     Haidong
  # Name: b, dtype: object
  
  print(df.iloc[0, 1:2])
  # Hunan    Yueyang
  # Name: a, dtype: object
  
  print(df.iloc[[0, 2], :])
  #     Hubei     Hunan Qinghai
  # a  Shiyan   Yueyang  Haibei
  # c  Hankou  Xiangtan   Haixi
  ```

- 布尔索引

  ```python
  df[df > 2]
  print(df[df.isin(["Changsha"])])
  #   Hubei     Hunan Qinghai
  # a   NaN       NaN     NaN
  # b   NaN  Changsha     NaN
  # c   NaN       NaN     NaN
  ```

  

#### 转换 numpy

```python
src = pd.read_csv(filename, sep="\s+", header=None)
src = src.values
row_num, col_num = src.shape
X = np.hstack((np.ones((row_num, 1)), src[:, 0: col_num - 1]))
Y = src[:, col_num - 1:col_num]
```





#### 数据清理命令

- 同时重命名所有列：`df.rename(columns = lambda x: x + '1')`
- 选择性地重命名列：`df.rename(columns = {'oldName': 'newName'})`
- 重命名所有的索引：`df.rename(index = lambda x: x + 1)`
- 按顺序重命名列：`df.columns = ['x', 'y', 'z']`。
- 检查是否存在空值，相应地返回一个布尔值arrray：`pd.isnull()`
- pd.isnull()的反向：`pd.notnull()`
- 删除所有包含空值的记录：`df.dropna()`
- 删除所有包含空值的列：`df.dropna(axis=1)`
- 用'n'代替每个空值：`df.fillna(n)`
- 要将series的所有数据类型转换为浮点数：`ser.astype(float)`
- 将所有数字1替换为'1'，将3替换为'3'：`ser.replace([1,2], ['one', 'two'])`

#### 分组、排序和过滤数据

- 返回列值的 groupby 对象：`df.groupby(colm)`
- 返回多列值的groupby对象：`df.groupby([colm1, colm2])`
- 按升序排序（按列）：`df.sort_values(colm1)`
- 要按降序排序（按列）：`df.sort_values(colm2, ascending=False)`
- 提取列值大于0.6的行：`df[df[colm] > 0.6]`

#### **其他**

- 将第一个DataFrame的行添加到第二个DataFrame的末尾：`df1.append(df2)`
- 将第一个DataFrame的列添加到第二个DataFrame的末尾：`pd.concat([df1,df2],axis=1)`
- 返回所有列的平均值：`df.mean()`
- 返回非空值的数量：`df.count()`

