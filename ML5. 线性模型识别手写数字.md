# 线性模型识别手写数字

<img src="https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210324194756.png" alt="mnist-3.0.1" style="zoom: 25%;" />

> MNIST 数据集 (Mixed National Institute of Standards and Technology database) 是美国国家标准与技术研究院收集整理的大型手写数字数据库,包含60,000个示例的训练集以及10,000个示例的测试集。

## 获取数据

http://yann.lecun.com/exdb/mnist/

```python
    memory = joblib.Memory('./view')
    fetch_openml_cached = memory.cache(fetch_openml)
    mnist_dataset = fetch_openml_cached('mnist_784', as_frame=True)
    mnist_dataset = pd.DataFrame(data=np.c_[mnist_dataset['data'], mnist_dataset['target']],
                                 columns=mnist_dataset['feature_names'] + ['target'])
    return mnist_dataset
```

打包为函数 `load_mnist`

## 查看分析

`pd.info()`

`pd.describe()`项的含义和相应的单个方法

- `count`：元素数
- `unique`：具有唯一（唯一）值的元素数
- `top`： 最频繁值（mode）
- `freq`：频率（出现次数）
- `mean`： 算术平均值
- `std`： 标准差
- `min`：最小值
- `max`： 最大值
- `50%`： 中位数（median）
- `25%`， ： 1/4 分位数， 3/4 分位数`75%`

```python
    print(mnist_dataset)
    mnist_dataset.info()

    print(mnist_dataset.columns)
    print(mnist_dataset.describe())
```

## 数据预处理

自己定义查看 numpy 数据

```python
def view_data(X, line_num=10):
    print("\nview_data\nshape:{0}, dim: {1}, dtype: {2}".format(X.shape, X.ndim, X.dtype))
    print(X[:line_num])
```

包装函数

```python
    X, y = mnist_matrix[:, 0:-1], mnist_matrix[:, -1]
    view_data(X, 1)
    view_data(y)
```

转换为合适的数据结构

```python
    mnist_matrix = np.array(mnist_matrix, dtype=float)
```
查看数据、可视化

```python
    digit = np.array(X[20], dtype=int)
    digit = digit.reshape(28, 28)
    plt.imshow(digit, cmap="binary")
    plt.show()
    
    plt.hist(y)
    plt.hist(y[:60000])
    plt.show()
```

归一化

```python
    X = mnist_matrix[:, 0:-1] / 255
    y = np.array(mnist_matrix[:, -1], dtype=int)
    view_data(X, 1)
    view_data(y)
```

## 划分数据集

```python
    train_threshold = 60000
    N = y.shape[0]
    X = np.hstack((np.ones((N, 1)), X))

    X_training = X[:train_threshold, :]
    y_training = y[:train_threshold]

    X_test = X[train_threshold:, :]
    y_test = y[train_threshold:]
```

## 训练

获取维度、进一步处理训练集、初始化模型、误差函数

批量损失函数是这样的：
$$
\mathrm{Lost}\left(h_{\theta}(x), y\right)=\left\{\begin{array}{ll}
-\log \left(h_{\theta}(x)\right) & y=1 \\
-\log \left(1-h_{\theta}(x)\right) & y=0
\end{array}\right.
$$
合并为这个式子：
$$
\mathrm{Lost}\left(h_{\theta}(x), y\right)=-y \log \left(h_{\theta}(x)\right)-(1-y) \log \left(1-h_{\theta}(x)\right)
$$

### 模型

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def hypothesis(W, X):
    return sigmoid(X @ W)

def logistic_regression(X, y, aim_digit):
    N, dim = X.shape[0], X.shape[1]
    W = np.zeros(dim, dtype=np.float32)

    tmp = aim_digit * np.ones(N, dtype=np.int32)
    y_new = np.array((y == tmp), dtype=np.int32)
    # view_data(y)

    error_func(W, X, y_new)
    result = optimize.minimize(fun=error_func, x0=W, args=(X, y), jac=gradient,  method='Newton-CG')
    return result.x
```



## 测试、评价模型

试一下

```python
    count = 5
    X_sample = X_test[:count, :]
    for i in range(count):
        digit = np.array(X_sample[i, 1:] * 255, dtype=int)
        digit = digit.reshape(28, 28)
        plt.imshow(digit, cmap="binary")
        plt.show()
    y_hat = predict(W, X_sample)
    print(y_hat == 1)
```

### 评价指标

准确率**(accuracy**)计算公式为：
$$
A C C=\frac{T P+T N}{T P+T N+F P+F N}
$$
**精确率**是针对我们预测结果而言的，它表示的是预测为正的样本中有多少是真正的正样本。你认为的正样本，有多少猜对了（猜的精确性如何）。那么预测为正就有两种可能了，一种就是把正类预测为正类(TP)，另一种就是把负类预测为正类(FP)，也就是
$$
P=\frac{TP}{TP+FP}
$$
而**召回率**表示的是样本中的正例有多少被预测正确了。或者说正样本有多少被找出来了（召回了多少）。那也有两种可能，一种是把原来的正类预测成正类(TP)，另一种就是把原来的正类预测为负类(FN)。
$$
R=\frac{TP}{TP+FN}
$$
在信息检索领域，精确率和召回率又被称为查准率和查全率，
$$
查准率＝检索出的相关信息量 / 检索出的信息总量 \\
查全率＝检索出的相关信息量 / 系统中的相关信息总量
$$
F-Measure 是 Precision 和 Recall 加权调和平均。F1 度量为
$$
F = \frac{2\times P\times R}{P+R}
$$



```python
def get_accuracy(W, X_test, y_test, aim_digit):
    N = y_test.shape[0]
    y_test = y_test == aim_digit
    y_hat = hypothesis(W, X_test) > 0.5
    view_data(y_test)
    view_data(y_hat)

    tmp = np.array(y_test == y_hat, dtype=np.int32)
    view_data(tmp)
    return np.mean(tmp) * 100
```

## 建立多分类器

利用 sigmoid 函数输出各个不同类型数字的概率，取最高概率，就得到了一个预测数字。有兴趣可以实现一下。

## scikit-learn

```python
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(solver="newton-cg")
logisticRegr.fit(X_training, y_training)
y_hat = logisticRegr.predict(X_test)
score = logisticRegr.score(X_test, y_test)
print(score)
```