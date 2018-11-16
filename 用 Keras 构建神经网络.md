# 用 Keras 构建神经网络

幸运的是，每次我们需要使用神经网络时，都不需要编写激活函数、梯度下降等。有很多包可以帮助我们，建议你了解这些包，包括以下包：

- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [Caffe](http://caffe.berkeleyvision.org/)
- [Theano](http://deeplearning.net/software/theano/)
- [Scikit-learn](http://scikit-learn.org/)
- 以及很多其他包！

在这门课程中，我们将学习 [Keras](https://keras.io/)。Keras 使神经网络的编写过程更简单。为了展示有多简单，你将用几行代码构建一个完全连接的简单网络。

我们会将在前几课学习的概念与 Keras 提供的方法关联起来。

该示例的一般流程是首先加载数据，然后定义网络，最后训练网络。

## 用 Keras 构建神经网络

要使用 Keras，你需要知道以下几个核心概念。

## 序列模型

```python
    from keras.models import Sequential

    #Create the Sequential model
    model = Sequential()
```

[keras.models.Sequential](https://keras.io/models/sequential/) 类是神经网络模型的封装容器。它会提供常见的函数，例如 `fit()`、`evaluate()` 和 `compile()`。我们将介绍这些函数（在碰到这些函数的时候）。我们开始研究模型的层吧。

## 层

Keras 层就像神经网络层。有全连接层、最大池化层和激活层。你可以使用模型的 `add()`函数添加层。例如，简单的模型可以如下所示：

```python
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten

    #创建序列模型
    model = Sequential()

    #第一层 - 添加有128个节点的全连接层以及32个节点的输入层
    model.add(Dense(128, input_dim=32))

    #第二层 - 添加 softmax 激活层
    model.add(Activation('softmax'))

    #第三层 - 添加全连接层
    model.add(Dense(10))

    #第四层 - 添加 Sigmoid 激活层
    model.add(Activation('sigmoid'))
```

Keras 将根据第一层自动推断后续所有层的形状。这意味着，你只需为第一层设置输入维度。

上面的第一层 `model.add(Dense(input_dim=32))` 将维度设为 32（表示数据来自 32 维空间）。第二层级获取第一层级的输出，并将输出维度设为 128 个节点。这种将输出传递给下一层级的链继续下去，直到最后一个层级（即模型的输出）。可以看出输出维度是 10。

构建好模型后，我们就可以用以下命令对其进行编译。我们将损失函数指定为我们一直处理的 `categorical_crossentropy`。我们还可以指定优化程序，稍后我们将了解这一概念，暂时将使用 `adam`。最后，我们可以指定评估模型用到的指标。我们将使用准确率。

```python
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ['accuracy'])
```

我们可以使用以下命令来查看模型架构：

```python
model.summary()
```

然后使用以下命令对其进行拟合，指定 epoch 次数和我们希望在屏幕上显示的信息详细程度。

然后使用fit命令训练模型并通过 epoch 参数来指定训练轮数（周期），每 epoch 完成对整数据集的一次遍历。 verbose 参数可以指定显示训练过程信息类型，这里定义为 0 表示不显示信息。

```python
model.fit(X, y, nb_epoch=1000, verbose=0)
```

*注意：在 Keras 1 中，nb_epoch 会设置 epoch 次数，但是在 Keras 2 中，变成了 epochs。*

最后，我们可以使用以下命令来评估模型：

```python
model.evaluate()
```

很简单，对吧？我们实践操作下。





## 练习

我们从最简单的示例开始。在此测验中，你将构建一个简单的多层前向反馈神经网络以解决 XOR 问题。

1. 将第一层设为 `Dense()` 层，并将节点数设为8，且 `input_dim` 设为 2。
2. 在第二层之后使用 softmax 激活函数。
3. 将输出层节点设为 2，因为输出只有 2 个类别。
4. 在输出层之后使用 softmax 激活函数。
5. 对模型运行 10 个 epoch。

准确度应该为 50%。可以接受，当然肯定不是太理想！在 4 个点中，只有 2 个点分类正确？**我们试着修改某些参数，以改变这一状况。例如，你可以增加 epoch 次数以及改变激活函数的类型。**如果准确率达到 75%，你将通过这道测验。能尝试达到 100% 吗？

首先，查看关于模型和层级的 Keras 文档。 Keras [多层感知器](https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py)网络示例和你要构建的类似。请将该示例当做指南，但是注意有很多不同之处。





##### 

- [network.py](https://classroom.udacity.com/nanodegrees/nd101-cn-advanced/parts/484efacf-1085-4481-9218-982074a4627c/modules/675d02b4-7881-4c86-8df8-7a6b3e12bbce/lessons/7c042ed0-08e7-4138-ad78-1a8d671d5da5/concepts/7c3d9b4c-206d-4430-9459-c78357181d9a#)
- [network_solution.py](https://classroom.udacity.com/nanodegrees/nd101-cn-advanced/parts/484efacf-1085-4481-9218-982074a4627c/modules/675d02b4-7881-4c86-8df8-7a6b3e12bbce/lessons/7c042ed0-08e7-4138-ad78-1a8d671d5da5/concepts/7c3d9b4c-206d-4430-9459-c78357181d9a#)







import numpy as np

from keras.utils import np_utils

import tensorflow as tf

\# Using TensorFlow 1.0.0; use tf.python_io in later versions

tf.python.control_flow_ops = tf



\# Set random seed

np.random.seed(42)



\# Our data

X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')

y = np.array([[0],[1],[1],[0]]).astype('float32')



\# Initial Setup for Keras

from keras.models import Sequential

from keras.layers.core import Dense, Activation

\# One-hot encoding the output

y = np_utils.to_categorical(y)



\# Building the model

xor = Sequential()



\# Add required layers

\# xor.add()



\# Specify loss as "binary_crossentropy", optimizer as "adam",

\# and add the accuracy metric

\# xor.compile()



\# Uncomment this line to print the model architecture

\# xor.summary()













- 重设练习
- 测试答案
- 提交答案





### 新技巧

我们的准确率达到了 75%，甚至会达到 100%，但是并不轻松！

这也暗示了在现实生活中，神经网络训练起来有点难。解决有 4 个数据点的简单 XOR 问题就需要一个庞大的架构！并且我们知道（理论上）具有 2 个隐藏节点的 2 层网络可以做到。

现在我们尝试一项任务。回到测验，并执行以下步骤：

- 将架构中第一个层，节点数改为64
- 加一个节点数为8的全链接层
- 第一个激活函数设为 `relu`（我们稍后将学习这一概念）
- 将 epoch 次数设为 100。

现在点击`测试运行`。准确率是多少？像我一样达到 100% 了吗？很神奇，对吧？似乎 relu 很有用。

在下面的几个部分，我们将学习几个类似的训练优化技巧。