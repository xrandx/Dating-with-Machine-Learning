## 第 9 课 NumPy 之 ndarray

NumPy 中文文档：https://www.numpy.org.cn/reference/

ndarray 是一个同类型数据的多维容器，我们很多操作都需要使用它。

### 课程纲要

- 创建（增）
  - 通过 Python 内置数据类型：`np.array([0, 1, 2, 3])`, `np.array([[0, 1, 2], [3, 4, 5]])`
  - 通过 np 的方法：`np.arange()`, `np.linspace()`, `np.ones()`, `np.zeros()`,  `np.diag()`,  `np.random.randint(10, size=6)`, `np.linspace(1, 6, 6) `, `np.arange(15).reshape((3, 5))`等

- 索引（查）

  - 切片

    ```python
    a = np.linspace(0, 9, 10)
    a[2:8]
    ```

  - 花式索引，通过整型数组，它总是将数据复制到新数组

    ```
    a = np.linspace(0, 9, 10)
    a[[2, 3 ,3 ,3]]
    a[[9, 7]] = -100
    
    arr = np.arange(32).reshape((8, 4))
    
    a[(2, 2)]
    arr[[1, 2], [0, 1]]
    arr[[1, 3, 2]][:,[0, 3,  2]]
    
    arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]
    ```

  - 布尔掩码( boolean masks)

    ```python
    a = np.arange(0, 100, 10)
    print(a)
    mask = a % 7 == 0
    print(mask)
    
    mask = (a % 3 == 0) & (a % 5 == 0)
    
    print(a[mask])
    
    ```

- 数组运算([文档](https://www.numpy.org.cn/reference/routines/math.html#%E6%8C%87%E6%95%B0%E5%92%8C%E5%AF%B9%E6%95%B0))

  - 数乘

    ```python
    arr = np.arange(16).reshape((4, 4))
    2 * arr
    ```

  - 点乘 

    ```
    a = np.array([[1,2], [3,4]])
    b = np.array([[5,6], [7,8]])
    np.dot(a.T, a)
    ```

  - 转置 `arr.T`

  - e^n `np.exp(arr)`

  - 绝对值 `np.abs(arr)`

  - 三角函数 `np.cos(arr)`

### 作业

- 尝试使用 `arange`, `linspace`, `ones`, `zeros`, `eye` 和 `diag` 函数创建数组。
- 在使用随机数之前设置随机数种子(seed)。
- 将数组保存为文件，然后从文件中恢复。[文档](https://www.numpy.org.cn/reference/routines/io.html#%E6%96%87%E6%9C%AC%E6%96%87%E4%BB%B6)
