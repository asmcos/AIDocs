#### LSTM
长短期记忆（Long short-term memory, LSTM）是一种特殊的RNN，主要是为了解决长序列训练过程中的梯度消失和梯度爆炸问题。简单来说，就是相比普通的RNN，LSTM能够在更长的序列中有更好的表现。

#### 例子代码
[战略家上的例子](http://www.zhanluejia.net.cn/zlj/question.html?questionid=5eddf8f9a4e0175b18806c73)

模拟了sin函数和股票预测

#### LSTM解决了RNN的以下问题
* 梯度消失会导致我们的神经网络中前面层的网络权重无法得到更新，也就停止了学习。
* 梯度爆炸会使得学习不稳定， 参数变化太大导致无法获取最优参数。
* 在深度多层感知机网络中，梯度爆炸会导致网络不稳定，最好的结果是无法从训练数据中学习，最坏的结果是由于权重值为NaN而无法更新权重。
* 在循环神经网络（RNN）中，梯度爆炸会导致网络不稳定，使得网络无法从训练数据中得到很好的学习，最好的结果是网络不能在长输入数据序列上学习。

#### 参考文档
1. https://zhuanlan.zhihu.com/p/32085405

LSTM 如何解决RNN带来的梯度消失问题

2. https://zhuanlan.zhihu.com/p/136223550

3. https://zhuanlan.zhihu.com/p/44163528

lstm gru 预测股票

4. https://www.kaggle.com/dpamgautam/stock-price-prediction-lstm-gru-rnn