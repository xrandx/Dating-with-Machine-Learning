## 第 13 课 Pandas (2)

### 课程纲要

#### 索引

- loc——通过行标签索引行数据

  ```python
  data = [[1,2,3],[4,5,6]]
  index = [0,1]
  columns=['a','b','c']
  df = pd.DataFrame(data=data, index=index, columns=columns)
  print df.loc[1]
  '''
  a    4
  b    5
  c    6
  '''
  ```

- iloc 在index的位置上进行索引，不包括end.

  ```python
  # 选择第1行数据
  df.iloc[0]
  ```

  

- ix 先在index的标签上索引，索引不到就在index的位置上索引(如果index非全整数),不包括end.

#### 转换 numpy

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

- 返回列值的groupby对象：`df.groupby(colm)`
- 返回多列值的groupby对象：`df.groupby([colm1, colm2])`
- 按升序排序（按列）：`df.sort_values(colm1)`
- 要按降序排序（按列）：`df.sort_values(colm2, ascending=False)`
- 提取列值大于0.6的行：`df[df[colm] > 0.6]`