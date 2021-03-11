# 模型评估与选择（数据集构建）

## 机器学习过程

回顾高一的运动学实验，探索运动学公式:

$$
x = \frac{1}{2}at^2
$$
把带有滑轮的长木板平放在实验桌上，把滑轮伸出桌面，把打点计时器固定在长木板上没有滑轮的一端，并把打点计时器连接在电源上。此时 $a = g = 10  m/s^2$



```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 20, 20)
y = 0.5 * 10 * (x**2)
trainning_set = y + np.random.randn(y.shape[-1]) * 2.5
plt.plot(x, trainning_set, 'gx')


plt.show()
```


![png](https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210311132510.png)
    



训练集 $X = \{(x_i, y_i)\}$ 这里等于 $X = \{(0, 0), (1, 20.6), (3, 45.2), \cdots \}$

测试集 

假设空间：一元一次函数，一元 n 次函数，牛顿的假设-一元二次函数




```python
def poly_format(coeff):
    fmt = ["%.2f" % (coeff[-1])]
    cnt = -1
    for i in reversed(coeff):
        cnt += 1
        if cnt == 0:
            continue
        fmt.append("%.2fx^%d + " % (i, cnt))
    fmt = reversed(fmt)
    return "".join(fmt)


```


```python
coeff = np.polyfit(x, trainning_set, 2)
print(poly_format(coeff))

poly2 = np.polyval(coeff, x)
plt.plot(x, trainning_set, 'gx')
plt.plot(x, poly2, 'b')
plt.show()
```

    5.01x^2 + -0.22x^1 + 0.64




![png](https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210311132518.png)
    



```python
coeff = np.polyfit(x, trainning_set, 1)
print(poly_format(coeff))
poly1 = np.polyval(coeff, x)
plt.plot(x, trainning_set, 'gx')
plt.plot(x, poly1, 'b')
plt.plot(x, test_set, 'rx')
plt.show()
```

    100.03x^1 + -315.95




![png](https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210311132521.png)
    



```python
coeff = np.polyfit(x, trainning_set, 3)
print(poly_format(coeff))
poly1 = np.polyval(coeff, x)
plt.plot(x, trainning_set, 'gx')
plt.plot(x, poly1, 'b')
plt.plot(x, test_set, 'rx')

plt.show()
```

    -0.00x^3 + 5.08x^2 + -0.73x^1 + 1.37




![png](https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210311132524.png)
    


## 模型归纳偏好

特化与泛化

没有免费的午餐：https://www.leiphone.com/news/201907/jswawIEtorcAYvrB.html

奥卡姆剃刀原则：如无必要，勿增实体

## 误差

我们将学习器对样本的实际预测结果与样本的真实值之间的差异成为：误差（error）。

 - 在训练集上的误差称为训练误差（training error）或经验误差（empirical error）。
 - 在测试集上的误差称为测试误差（test error）。
 - 学习器在所有新样本上的误差称为泛化误差（generalization error）。

显然，我们希望得到的是在新样本上表现得很好的学习器，即泛化误差小的学习器。因此，我们应该让学习器尽可能地从训练集中学出普适性的“一般特征”，这样在遇到新样本时才能做出正确的判别。然而，当学习器把训练集学得“太好”的时候，即把一些训练样本的自身特点当做了普遍特征；同时也有学习能力不足的情况，即训练集的基本特征都没有学习出来。我们定义：

 - 学习能力过强，以至于把训练样本所包含的不太一般的特性都学到了，称为：过拟合（overfitting）。
 - 学习能太差，训练样本的一般性质尚未学好，称为：欠拟合（underfitting）。

## 训练集与测试集的构建方法

我们希望用一个“测试集”的“测试误差”来作为“泛化误差”（因为不可能知道）的近似，因此我们需要对初始数据集进行有效划分，划分出互斥的“训练集”和“测试集”。下面介绍几种常用的划分方法：


### 留出法

将数据集D划分为两个互斥的集合，一个作为训练集 $S$，一个作为测试集$T$ ，满足 $D=S∪T$ 且 $S∩T=∅$

常见的划分为：大约2/3-4/5的样本用作训练，剩下的用作测试。需要注意的是：训练/测试集的划分要尽可能保持数据分布的一致性，以避免由于分布的差异引入额外的偏差，常见的做法是采取分层抽样。同时，由于划分的随机性，单次的留出法结果往往不够稳定，一般要采用若干次随机划分，重复实验取平均值的做法。

### 交叉验证法

 将数据集 $D$ 划分为两个互斥的集合，一个作为训练集 $S$，一个作为测试集 $T$，满足 $D=S∪T且S∩T=∅$ ，常见的划分为：大约$2/3-4/5$的样本用作训练，剩下的用作测试。需要注意的是：训练/测试集的划分要尽可能保持数据分布的一致性，以避免由于分布的差异引入额外的偏差，常见的做法是采取分层抽样。同时，由于划分的随机性，单次的留出法结果往往不够稳定，一般要采用若干次随机划分，重复实验取平均值的做法。

 ### 自助法
我们希望评估的是用整个D训练出的模型。但在留出法和交叉验证法中，由于保留了一部分样本用于测试，因此实际评估的模型所使用的训练集比D小，这必然会引入一些因训练样本规模不同而导致的估计偏差。留一法受训练样本规模变化的影响较小，但计算复杂度又太高了。“自助法”正是解决了这样的问题。

自助法的基本思想是：给定包含m个样本的数据集D，每次随机从D 中挑选一个样本，将其拷贝放入D'，然后再将该样本放回初始数据集D 中，使得该样本在下次采样时仍有可能被采到。重复执行m 次，就可以得到了包含m个样本的数据集D'。 

 ### 调参
大多数学习算法都有些参数(parameter) 需要设定，参数配置不同，学得模型的性能往往有显著差别，这就是通常所说的"参数调节"或简称"调参" (parameter tuning)。

学习算法的很多参数是在实数范围内取值，因此，对每种参数取值都训练出模型来是不可行的。常用的做法是：对每个参数选定一个范围和步长λ，这样使得学习的过程变得可行。

## ML1答案

1. BCD
2. 特征选择错了

答案可见 ML3 视频开头