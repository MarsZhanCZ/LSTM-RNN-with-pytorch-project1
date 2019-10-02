import numpy as np
import tushare as ts
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#data_close = ts.get_k_data('000001', start='2018-01-01', index=True)['close'].values  # 获取上证指数从20180101开始的收盘价的np.ndarray
data_close = ts.get_hist_data('600036', start='20170101', end='20190830')['close'].values
#data_close = ts.get_hist_data('600770', start='20180101', end='20190830')['close'].values
data_close = data_close.astype('float32')  # 转换数据类型
print(type(data_close))
#data_close = data_close[::-1]

# 将价格标准化到0~1
max_value = np.max(data_close)
min_value = np.min(data_close)
data_close = (data_close - min_value) / (max_value - min_value)

'''
把K线数据进行分割，每 DAYS_FOR_TRAIN 个收盘价对应 1 个未来的收盘价。例如K线为 [1,2,3,4,5]， DAYS_FOR_TRAIN=3，那么将会生成2组数据：
第1组的输入是 [1,2,3]，对应输出 4；
第2组的输入是 [2,3,4]，对应输出 5。
然后只使用前70%的数据用于训练，剩下的不用，用来与实际数据进行对比。
'''

DAYS_FOR_TRAIN = 10


def create_dataset(data, days_for_train=5) -> (np.array, np.array): #->用于指示函数返回的类型
    """
        根据给定的序列data，生成数据集

        数据集分为输入和输出，每一个输入的长度为days_for_train，每一个输出的长度为1。
        也就是说用days_for_train天的数据，对应下一天的数据。

        若给定序列的长度为d，将输出长度为(d-days_for_train+1)个输入/输出对
    """
    dataset_cmp_ud=[[0,0]]
    last = -1;#初始值
    print('data长度：',len(data))
    for i in range(10, len(data)-1):#0,1,2...len-1
        if data[i+1] > data[i]:
            dataset_cmp_ud.append([0, 1])
        elif data[i+1] < data[i]:
            dataset_cmp_ud.append([1, 0])
        else:
            dataset_cmp_ud.append([0, 1])
    print(dataset_cmp_ud)
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train):
        _x = data[i:(i + days_for_train)]#取第i到(i + days_for_train)-1列数据
        dataset_x.append(_x)#[[d0:d4],[d1:d5],[d2:d6]...]
        #dataset_y.append(data[i + days_for_train])#[d5,d6,d7...]
        dataset_y = dataset_cmp_ud
    return (np.array(dataset_x), np.array(dataset_y))


dataset_x, dataset_y = create_dataset(data_close, DAYS_FOR_TRAIN)

# 划分训练集和测试集，70%作为训练集
train_size = int(len(dataset_x) * 0.8)

train_x = dataset_x[:train_size]
train_y = dataset_y[:train_size]

# 将数据改变形状，RNN 读入的数据维度是 (seq_size, batch_size, feature_size)
train_x = train_x.reshape(-1, 1, DAYS_FOR_TRAIN)
train_y = train_y.reshape(-1, 1, 2)

# 转为pytorch的tensor对象
train_x = torch.from_numpy(train_x)#torch.Size([129, 1, 10])
print(train_x.size())
train_y = torch.from_numpy(train_y)

#定义网络、优化器、Loss函数
class LSTM_Regression(nn.Module):
    """
        使用LSTM进行回归

        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size, output_size=2, num_layers=2, num_class=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.classifier = nn.Linear(hidden_size, num_class)

    def forward(self, _x):
        #print('train_x的维度是：', _x.shape)
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        #print('输出的维度是s, b, h:', s, b, h)
        x = x.view(s * b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        #out =  F.log_softmax(x, dim=-1)
        #print(out)

        #x输出的是预测值，out_classifier输出的是三分类
        #out_classifier = x
        #out_classifier = out_classifier[-1, :, :]
        #out_classifier = self.classifier(out_classifier)
        #print(x, out_classifier)
        return x#, out_classifier


model = LSTM_Regression(DAYS_FOR_TRAIN, 16, output_size=2, num_layers=2, num_class=2)#如果是要多维输出则output_size=n

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)
loss_track = []

#训练
for i in range(2300):#epoch_times
    out = model(train_x)#等价于model.forward(train_x)
    #print(out.size())#torch.Size([129.1.1])
    #print('当前网络输出是：')
    #print(out)
    #print(train_y)

    loss = loss_function(out, train_y.float())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (i + 1) % 10 == 0:
        loss_for_plt = round(loss.item(), 5)
        loss_track.append(loss_for_plt)
        #print(loss_for_plt)

    if (i + 1) % 100 == 0:
        print('Epoch: {}, Loss:{:.5f}'.format(i + 1, loss.item()))


#测试
import matplotlib.pyplot as plt

model = model.eval()  # 转换成测试模式，pytorch会自动把batch_normalization和DropOut等函数固定住，不会取平均，而是用训练好的值

# 注意这里用的是全集 模型的输出长度会比原数据少DAYS_FOR_TRAIN(0-DAYS_FOR_TRAIN日是没有预测的) 填充使长度相等再作图
dataset_x = dataset_x.reshape(-1, 1, DAYS_FOR_TRAIN)  # (seq_size, batch_size, feature_size)
dataset_x = torch.from_numpy(dataset_x)

pred_test = model(dataset_x)  # 全量训练集的模型输出 (seq_size, batch_size, output_size)
pred_test = pred_test.data.numpy()#pred_test = pred_test.view(-1).data.numpy()
#print(pred_test)
#pred_test = np.concatenate((np.zeros(DAYS_FOR_TRAIN), pred_test))  # 填充0 使长度相同,concatenate是矩阵拼接
print(len(pred_test), len(dataset_y))
#assert len(pred_test) == len(dataset_y)#检查条件，不符合就终止程序

stat_for_def_total=0
stat_for_def_once=0
for i in range(0, len(dataset_y)):
    if (np.argmax(pred_test[i][0]) == (np.argmax(dataset_y[i]))):
        stat_for_def_total+=1

        if i > (0.7*len(dataset_x)):
            stat_for_def_once += 1
print('涨跌趋势2分类全集正确率：',stat_for_def_total/(len(dataset_y)+1))
print('涨跌趋势2分类全集正确率：', stat_for_def_once / ((len(dataset_y) + 1)*0.3))

'''
下面部分都是绘图，正确率还不知道怎么画图好…所以没有修改绘图部分的代码。
目前程序运行到这会自动停，幸好python是解释型语言不影响前面的运算。
'''
plt.plot(pred_test, 'r', label='prediction')
plt.plot(dataset_y, 'b', label='real')
plt.plot((train_size, train_size), (0, 1), 'g--')
plt.legend(loc='best')
plt.show()

plt.plot(loss_track, 'r',label='loss')
plt.legend(loc='best')
plt.show()
