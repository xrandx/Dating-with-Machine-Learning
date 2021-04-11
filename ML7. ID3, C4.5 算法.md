# ID3, C4.5 算法

<img src="https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210328170928.svg" alt="决策树" style="zoom: 67%;" />

## ID3 算法

信息熵
$$
\mathrm{H}(X) = \sum_i^np_ilog_2(1/p_i) = -\sum_i^np_ilog_2(p_i)
$$
条件熵
$$
\begin{aligned}

H(Y \mid X) 
&=\sum_{i=1}^{n} p\left(x_{i}\right) H\left(Y \mid X=x_{i}\right) \\
&=-\sum_{i=1}^{n} p\left(x_{i}\right) \sum_{j=1}^{m} p\left(y_{j} \mid x_{i}\right) \log _{2} p\left(y_{j} \mid x_{i}\right) \\
&=-\sum_{i=1}^{n} \sum_{j=1}^{m} p\left(x_{i}, y_{j}\right) \log _{2} p\left(y_{j} \mid x_{i}\right)

\end{aligned}
$$
信息增益
$$
G(X) = H(Y) - H(Y|X)
$$


$Y = 1$ 表示买了。$Y = 0$ 表示没买。

$X = 1$ 表示附近学校好。$X = 0$ 表示附近学校不好。

觉得附近学校好，其中买的人有 5 个，不买的个数为 6 个；

觉得附近学校好其中买的人有 1 个，不买的个数为 8 个；

可以得到概率：
$$
P(Y = 1|X = 1) = \frac{5}{11}\\
P(Y = 0|X = 1) = \frac{6}{11} \\
P(Y = 0 | X = 0) = \frac{8}{9} \\
P(Y = 1 | X = 1) = \frac{1}{9} \\
$$

各个条件熵：
$$
\begin{aligned}
H(Y=1|X = 1)  &= -\frac{5}{11}log_2(\frac{5}{11}) -\frac{6}{11}log_2(\frac{6}{11}) = 0.99 
\\
H(Y=1|X=0) &=  -\frac{1}{9}log_2(\frac{1}{9}) -\frac{8}{9}log_2(\frac{8}{9}) = 0.5\\
\end{aligned}
$$
按期望平均得到条件熵，算出 Y = 1 的信息熵：
$$
\begin{aligned}

&P(X = 1) =   \frac{11}{20}\\
&P(X =0) = \frac{9}{20}
\\ \\
&H(Y=1|X) =   \frac{11}{20} \times  0.99  +  \frac{9}{20} \times 0.5 = 0.77
\end{aligned}
$$

算出 Y = 1 的信息熵
$$
H(Y = 1) = -\frac{6}{20}log_2(\frac{6}{20}) -\frac{14}{20}log_2(\frac{14}{20}) = 0.88
$$
然后得出 X 事件的信息增益：
$$
G(X) = H(Y = 1) - H(Y = 1|X) = 0.88-0.77 = 0.11
$$

### 利用信息增益构建决策树

(案例出自西瓜书)

拿西瓜来说，他的样本属性可能是 $[色泽，瓜蒂，敲声，纹理,\dots]$，例如西瓜样本 

<img src="https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210411161337.png" alt="{3C0EB52A-E0E3-4D52-9B78-D62220062A5C}" style="zoom: 50%;" />

我们算出来所有属性的信息增益，D 是样本集合（如上图）：
$$
G(D，瓜蒂) = 0.143 \\
G(D，纹理) = 0.381 \\
G(D，脐部) = 0.289 \\
G(D，触感) = 0.006 \\
G(D，敲声) = 0.141
$$
此时，触感的信息增益最大，我们按照触感划分样本集合，得 D1, D2,  D3

<img src="https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210411162149.png" style="zoom:50%;" />
$$
G(D_1 ， 色泽) = 0.043\\ G(D_1 ，根蒂) = 0.458 \\ G(D_1 ，敲声) = 0.331 \\ G(D_1 ，脐部) = 0.458\\ G(D_1 ，触感) = 0.458
$$
……按照这种划分，我们就建立起了一棵决策树：

<img src="https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210411162915.png" alt="{02BA7EDD-3E25-481F-82EE-52CBA23D1367}" style="zoom: 67%;" />

ID3 算法缺点：

1. 连续特征无法在ID3运用。
2. ID3 采用信息增益大的特征优先建立决策树的节点，在相同条件下，取值比较多的特征比取值少的特征信息增益大，这对预测性能影响很大。
3. ID3算法对于缺失值的情况没有做考虑。
4. 没有考虑过拟合的问题。

## C4.5 算法

### 信息增益比

信息增益准则对取值数目较多的属性有所偏好，ID3 算法的作者 Quinlan 基于上述不足，对ID3算法做了改进，不直接使用信息增益，而使用信息增益比：
$$
R_G(D, A) = \frac{G(D, A)}{IV_A(D)}
$$
D 是样本集合，A 是样本的某个属性，分母是样本 D 关于的属性 A 的固有值 (Intrinsic Value)：
$$
IV_D(A) = -\sum^n_i \frac{|D_i|}{|D|} log_2\frac{|D_i|}{|D|}
$$
属性 A 的某个取值越多，IV 的值就越大：
$$
IV_D(触感) = 0.874 (V = 2) \\ IV_D(色泽) = 1.580 (V = 3) \\ IV_D(编号) = 4.088 (V = 17)
$$
### 连续特征离散化

假设属性 A 的所有取值有 m 个，从小到大排列为 $a_1,a_2,...,a_m$ ，则 C4.5 取相邻两样本值的平均数，一共取得 $m-1$ 个划分点，其中第 $i$ 个划分点 $T_i$表示为：$T_i=\frac{ai+ai+1}{2}$。对于这 $m−1$ 个点，分别计算以该点作为二元分类点时的信息增益。选择信息增益最大的点作为该连续特征的二元离散分类点。

比如取到的增益最大的点为 $a_t$ ,则小于 $a_t$ 的值为类别 1，大于 $a_t$ 的值为类别 2，这样我们就做到了连续特征的离散化。要注意的是，与离散属性不同的是，如果当前节点为连续属性，则该属性后面还可以参与子节点的产生选择过程。

### 缺失值处理

1. 在样本某些特征缺失的情况下选择划分的属性
2. 选定了划分属性，对于在该属性上缺失特征的样本的处理

……

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


def main():
    iris = load_iris()
    print(iris["feature_names"])

    X = iris.data[:, 2:]  # petal length and width
    y = iris.target

    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(X, y)


if __name__ == '__main__':
    main()
```

