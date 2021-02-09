## 第 12 课 Pandas (1)

`Pandas`这个名称来源于panel data（面板数据），是一个开源的，BSD许可的库，为 Python 编程语言提供高性能，易于使用的数据结构和数据分析工具。

### 资源

官方文档：https://www.pypandas.cn/docs/

### 课程纲要

- 开始
  - Pandas 用来干嘛
    - 方便处理数据清洗；
    - 能处理时间序列数据和非时间序列数据；
    - 提供了表格结构，可以用 numpy 来计算，实现了从日常表格数据到高性能数组的跨越。
  - `import pandas as pd`
  
- **Series**
  
- 直观理解：一维的对象，像是一个数据按创建先后排序的字典，也可以像数组一样访问，包括索引和数据。
  
  - 初始化（增）
    - `obj = Series([1, 2, -3, 5])`
    - 可从字典初始化 `Series`     
    - `obj = Series([1, 2, -3, 5], index=['a', 'b', 'c', 'f'])`
    - `obj = pd.Series(5, index=['a', 'b', 'c', 'f'])`
    
  - 查找（查）
  - `obj.values` 获取值 `values`
  
      ```python
      for val in obj1.values:
          print(val)
    ```
  
  - `obj.index` 获取索引 `index`
  
      ```python
      print(obj1.index)
    ```
  
  - `'b' in obj2` 是否存在 index
  
  - `obj[0]` 按位置查找
  
  - `obj[index]`  `obj[['a', 'b']]` 按索引查找
  
  - `obj[:5]` 切片表达式
  
    - 查不到会出现`NaN，即“非数字”（not a number）`
  
- 赋值（改）
  
  - 注意：Pandas 用 `NaN`（Not a Number）表示缺失数据，`Series` 对象可以包含不同类型的数值。
  
    - 用`[]`修改`values`
  
      ```python
      obj1 = pd.Series(np.arange(4), index=['a', 'b', 'c', 'f'])
      obj1['a', 'c'] = "modified"
      obj1[:2] = "modified"
      obj1["new"] = "modified"
      ```
    
  - 改变 `index`
    
      ```python
      obj1 = pd.Series(np.arange(4), index=['a', 'b', 'c', 'f'])
      print(obj1)
    obj1.index = ['x', 'y', 'z', '1']
    ```

  
  - 改变 `name`
    
      ```python
      obj1.name = "text"
	  ```
  
- 运算
  
    - 自动对齐
    
      ```python
      obj1 = pd.Series(1, index=['a', 'b', 'c', 'f'])
      obj2 = pd.Series(1, index=['a', 'b', 'c', 'd'])
    print(obj1 + obj2)
      ```
  
- **DataFrame**

  - 直观理解：二维的对象，含有一组有序的列。它类似于字典，该字典里每一个元素是 `Series` 。

  - 初始化（增）

    - 通过字典创建(Series 字典、字典、多维数组字典、列表字典、元组字典)

      ```python
      src = {
          'Hubei': ['Shiyan', 'Xiaogan', None],
          'Hunan': ['Yueyang', 'Changsha', 'Xiangtan'],
          'Qinghai': ['Haibei', 'Haidong', 'Haixi']
      }
      df = pd.DataFrame(src)
      # df = pd.DataFrame.from_dict(src)
      print(df)
      #       Hunan    Hubei  Qinghai
      #0   Yueyang   Shiyan   Haibei
      #1  Changsha  Xiaogan  Haidong
      #2  Xiangtan     None    Haixi
      
      ```

    - 改变顺序，添加索引

      ```python
      df = pd.DataFrame(src, columns=["Hunan", 'Hubei', 'Qinghai'])
      df = pd.DataFrame(src, columns=["Hunan", 'Hubei', 'Qinghai'], index=['one', 'two', 'three'])
      ```

    - 增加新列

      ```python
      df['isChenzhou'] = df['Hunan'] == "Changsha"
      print(df)
      #           Hunan    Hubei  Qinghai  isChenzhou
      # one     Yueyang   Shiyan   Haibei       False
      # two    Changsha  Xiaogan  Haidong        True
      # three  Xiangtan     None    Haixi       False
      ```

  - 查找（查）

    - 根据列名访问某一列 Series （输出索引相同的 Series）

      ```python
      df.Hunan
      df["Hunan"]
      
      # 结果都是：
      # one       Yueyang
      # two      Changsha
      # three    Xiangtan
      # Name: Hunan, dtype: object
      ```

    - 通过 Series 的索引访问具体元素

      ```python
      df["Hunan"].one
      df.Hunan.one
      df["Hunan"]["one"]
      df.Hunan["one"]
      #	结果都是：
      # 	Yueyang
      ```

  - 赋值（改）

    - 修改某列为相同数值

      ```python
      df['Hunan'] = 1
      print(df)
      #        Hunan    Hubei  Qinghai
      # one        1   Shiyan   Haibei
      # two        1  Xiaogan  Haidong
      # three      1     None    Haixi
      ```

    - 修改某列的数据，要长度一致。

      ```python
      df['Hunan'] = [2 * i for i in range(3)]	#	也可以用 np.arange 等
      print(df)
      #        Hunan    Hubei  Qinghai
      # one        0   Shiyan   Haibei
      # two        2  Xiaogan  Haidong
      # three      4     None    Haixi
      ```

    - 用 Series 精确匹配值

      ```python
      df['Hunan'] = pd.Series({
          'one': "Chenzhou",
          'two': "Hengyang",
          'three': pd.NA
      })
      #           Hunan    Hubei  Qinghai
      # one    Chenzhou   Shiyan   Haibei
      # two    Hengyang  Xiaogan  Haidong
      # three      None     None    Haixi
      ```

  - 删除（删）

    - 删除一列元素（对 Series 的修改会反应到 df 上）

      ```python
    del df['Hubei']
      print(df)
      #           Hunan  Qinghai
      # one     Yueyang   Haibei
      # two    Changsha  Haidong
      # three  Xiangtan    Haixi
      ```
  
  - 运算

    - 转置

      ```python
    print(df.T)
      #              one       two     three
      # Hunan    Yueyang  Changsha  Xiangtan
      # Hubei     Shiyan   Xiaogan      None
      # Qinghai   Haibei   Haidong     Haixi
      ```
  
    - 重新索引