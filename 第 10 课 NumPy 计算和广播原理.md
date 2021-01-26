## 第 10 课 NumPy 计算和广播原理

### 前情提要

#### 第 9 课补

- `arr[2, :]`

- `np.ix_()`

    ```python
    import numpy as np

    arr2 = np.arange(32).reshape((8,4))
    print(arr2)

    arr2[np.ix_([1,5,7,2],[0,3,1,2])]

    arr2[np.ix_([1,5,7,2],[0,1,2])]
    ```

#### 第 9 课答案

1. 略

2. 3. [numpy.random.randint](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html)

       ```python
       np.random.seed(3)
       x1 = np.random.randint(low=1, high=10, size=100)
       np.save("save.npy", x1)
       
       x1 = np.load("save.npy")
       print(x1)
       ```
       
       


### 课程纲要

- 数学计算

  - 逻辑运算`np.where` `np.all()` `np.any()`

    ```python
    arr = np.random.rand(2, 3)
    result = np.where(arr > 0.5, 1, 0)
    #	condition ? arr1: arr2
    arr = np.random.rand(2, 3)
    result = np.all(arr > 0.1)
    ```

  - 集合运算：[文档](https://www.numpy.org.cn/reference/routines/set.html#%E6%93%8D%E4%BD%9C%E9%9B%86%E5%90%88-set-routines)

    ```python
    arr = np.array([2, 2, 3, 3, 1])
    result = np.unique(arr)
    ```

  - 线性代数：[numpy.linalg 库文档](https://www.numpy.org.cn/reference/routines/linalg.html#%E7%9F%A9%E9%98%B5%E5%92%8C%E5%90%91%E9%87%8F%E7%A7%AF)

    - `np.trace` `np.inner()` 

    - `linalg.qr` `linalg.svd`

      ```python
      arr = np.array([[2, 2, 3, 3], [4, 3, 2, 1]])
      U, s, V = np.linalg.svd(arr)
      ```

  - 统计运算

    - 求和 `arr.sum(axis=0)`

      ```python
    arr = np.random.rand(2, 3)
      result = np.sum(arr)
      
      result = np.sum(arr > 0.5)	#	根据条件求真值的和
      ```

    - 最值 `arr.max()` `arr.min()` `arr.argmin() ` `arr.argmax()` [文档](https://www.numpy.org.cn/reference/routines/sort.html#searching)

      ```python
    arr = np.random.rand(2, 3)
      result = np.max(arr)
      
      arr = np.random.rand(2, 3)
      result = np.argmin(arr)
      ```

    - 算术平均数 `np.mean()`
    
      ```python
      arr = np.random.rand(2, 3)
      result = np.mean(arr)
      ```
    
    - 标准差、方差 `np.std()` `np.var()`

- Numpy 的形状操作

  - 添加维度

    ```python
    arr = np.array([1, 2, 3])
    result = arr[np.newaxis, :, ]
    print(arr)
    print(result)
    ```

  - 改变维度个数和大小  `np.resize() ` `np.reshape()` ，前者改变源数组，后者不会。

    ```python
    arr = np.random.random((4, 4))
    print(arr)
    arr.resize((2, 3))
    print(arr)
    arr.resize((1))
    print(arr)
    ```

- NumPy 的广播原理

  - 维度和维度大小

  - 广播（broadcasting)

    - 什么是广播

      ```python
      arr = np.ones((2, 4))
      arr2 = 1
      print(arr)
      print(arr2)
      print(arr + arr2)
      ```

    - 规则1：数组维度和大小，从后往前有连续的相同部分

      ```python
      arr = np.ones((2, 4, 4))
      arr2 = np.ones((4, 4))
      ```

    - 规则2：不相同的部分维度大小为1

      ```python
      arr = np.zeros((2, 4, 4))
      arr2 = np.ones((1, 1))
      ```

- Matplotlib

  - 画一个三角函数吧

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    
    X = np.linspace(-np.pi, np.pi, 100)
    COS, SIN = np.cos(X), np.sin(X)
    ax = np.zeros(100)
    plt.plot(COS)
    plt.plot(SIN)
    plt.plot(ax)
    plt.show()
    
    ```

