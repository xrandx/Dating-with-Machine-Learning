## 第 11 课 Matplotlib 画图

#### 官方文档

用户入门：https://matplotlib.org/users/index.html

pyplot 部分：https://matplotlib.org/tutorials/introductory/pyplot.html

样例：https://matplotlib.org/3.3.3/gallery/index.html

### 课程大纲

- API 是什么？Application Programming Interface

- figure, subplot

  - 添加 subplot

    ```python
    x = np.linspace(-np.pi, np.pi, 100, endpoint=True)
    y = np.cos(x)
    fig = plt.figure()
    fig.add_subplot(2, 2, 3)
    plt.plot(x, y)
    plt.show()
    ```

  - 批量生成 subplot

    ```python
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(x, y)
    y = np.sin(x)
    axes[1, 1].plot(x, y)
    plt.show()
    ```

  - 调整间距

    ```python
    plt.subplots_adjust(wspace=2, hspace=2)
    ```

- 改变样式

  - 线型

    ```python
    plt.plot(x, y, linewidth=10, color="blue", linestyle="-") 
    axes[0, 0].plot(x, y, linewidth=10, color="blue", linestyle="-") 
    ```

  - 坐标轴的范围、刻度、刻度名、坐标名

    ```python
    plt.xlim([-2 * np.pi, 2 * np.pi])
    plt.xlim(x.min() * 1.1, x.max() * 1.1)
    plt.ylim(y.min() * 1.1, y.max() * 1.1)
    plt.xticks(np.arange(-2 * np.pi, 2 * np.pi, np.pi / 2))
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                   [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$+\frac{\pi}{ 2}$', r'$+\pi$'])
    
    ```

  - 设置坐标轴

    ```python
    #   latex 公式
    ax = plt.gca()
    
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    #   bottom left
    ax.xaxis.set_ticks_position('bottom')
    #   position[0] should be one of 'outward', 'axes', or 'data'
    ax.spines['bottom'].set_position(("data", 0))
    
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(("data", 0))
    ```

  - 添加图例

    ```python
    plt.plot(x, y, linewidth=1, color="blue", linestyle="-", label="Cos")
    y = np.sin(x)
    plt.plot(x, y, linewidth=1, color="red", linestyle="-", label="Sin")
    plt.legend(loc='upper left')
    ```

  - 标注数据

    ```python
    plt.text(0, 1, "max_value", fontsize=15)
    ```

- 不同种类的图形

  - 散点图

    ```python
    x = np.random.randn(100, 100)
    y = np.random.randn(100, 100)
    plt.scatter(x, y)
    plt.show()
    ```

  - 条形图（横放条形图，竖放条形图，并列条形图）、饼图

    ```python
    name_list = np.array(['Monday', 'Tuesday', 'Friday', 'Sunday'])
    num_list = [1.5, 0.6, 7.8, 6]
    plt.bar(range(len(num_list)), num_list, color='black', tick_label=name_list)
    plt.show()
    
    Z = np.random.uniform(0, 1, 20)
    plt.pie(Z)
    ```

  - 等高线

  - 箭头图

- 保存文件

  ```python
  plt.savefig('fig.png')
  ```

- 查看文档

  - ```python
    import matplotlib.pyplot as plt
    help(plt.plot)
    ```