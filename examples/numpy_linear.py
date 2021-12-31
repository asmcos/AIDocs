#
# 使用numpy 实现线性回归
# y = w * x + b
#

import numpy as np
import matplotlib.pyplot as plt


#
# 例子1.一个普通的线性回归
#
def predict(x, w, b):
    return w * x + b

#随机一个w,b 
b = np.random.normal(0,1)
w = np.random.normal(0,1)

#100个数字
x = np.arange(100)
#产生 100个随机数，在0-20之间
actual_y = np.random.uniform(0,20,len(x))

# 未建立关联

plt.plot(predict(x, w, b),c='r',label="pred")
plt.plot(actual_y,c='b',label="actual_y")
plt.legend()
plt.figure()


#
# 例子2. 
#

#计算损失
def loss(pred_y,y):
    v = pred_y - y
    return np.sum(v * v) / len(y)

# 手动对w求偏导
def partial_w(x,pred_y,y):
    return 2 / len(pred_y) * np.sum((pred_y-y)*x)

# 手动对b求偏导
def partial_b(x, pred_y,y):
    return 2 / len(pred_y) * np.sum(pred_y-y)

learning_rate = 0.0001
result = []
for i in range(60):
    pred_y = predict(x,w,b)
    lss = loss(pred_y,actual_y)
    result.append(lss)
    w = w - learning_rate * partial_w(x, pred_y, actual_y)
    b = b - learning_rate * partial_b(x, pred_y, actual_y)

plt.plot(x,pred_y, c='y',label="pred_y")
plt.legend()
plt.show()
