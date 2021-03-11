# 机器学习入门

## 课程纲要

### 什么是机器学习？

Improving some performance measure with experience computed from data. 

机器从数据中总结经验。

### 什么时候用机器学习？

- 事物存在某种潜在规律
- 人不能直接发现这种规律 （例如牛顿定律）
- 能获取大量数据

### 机器学习概念

- 输入空间：$\mathcal{X}$
- 输出空间：$\mathcal{Y}$
- 假设空间( hypothesis space )：$\mathcal{H}$ , 包含所有可能的 $f :\mathcal{X} \mapsto \mathcal{Y}$
- 所有记录的集合：数据集, $\mathcal{D}=\{\left(\mathbf {x_i},Y_i\right)|1\le i\le m\}$
- 一条记录( instance, sample ) $\mathbf{x_i}$
- 数据的特征或者属性 feature, attribute : $\mathbf{x_i} = \{x_1, x_2, \cdots, x_n \}$
- 训练集 
- 测试集

### 假设空间——机器学习的过程

<img src="https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210303185856.png" alt="new-pic-mine-e1572009324171" style="zoom:67%;" />

### 机器学习分类

 - 预测值为离散值或连续值的问题为：
   - 分类（classification）（上火问题，是否下雨）
   - 回归（regression）$\mathcal{R}$

 - 训练数据有标记信息的学习任务为：监督学习（supervised learning），分类和回归都属于监督学习。
 - 训练数据没有标记信息的学习任务为：无监督学习（unsupervised learning），常见的有聚类和关联规则。
 - 还有：batch learning, online learning, active learning, reinforcement Learning

### 为什么可以学习？

简要解释计算学习理论：

Ein(h)表示在**训练集样本**中，h(x)不与f(x)不相等的概率。即模型假设对样本（已知）的错误率。

Eout(h)表示**实际所有样本**中，h(x)与f(x)不相等的概率。即模型假设对真实情况（未知）的错误率。

霍夫丁不等式：
$$
P[|\nu-\mu|>\epsilon] \leq 2 e^{-2 \epsilon^{2} N}
$$
PAC

### 数据分析的一般流程

- 数据清理和格式化
- 探索性数据分析
- 特征工程和特征选择
- 基于性能指标比较几种机器学习模型
- 对最佳模型执行超参数调整
- 在测试集上评估最佳模型
- 解释模型结果
- 得出结论

<img src="https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210303185958.png" alt="13bb24f42e5bb98f4a9c15037e523d7d" style="zoom: 33%;" />

## 作业


### 选择题

#### 以下哪些问题适合用机器学习来解决?  

A.  判断今年是闰年还是平年

B. 判断银行能不能给某人开信用卡

C. 判断北京明天的天气

D. 估计北京西直门早高峰的人流量

E. 计算地球运行的轨道

### 问答题

#### 亚里士多德提出「物体下落的快慢是由物体本身的重量决定的」，他的错误出现在数据分析的哪一步？






