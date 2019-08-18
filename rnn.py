import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline#画一个作图框

plt.figure(figsize=(8,5))#t图片尺寸
#plt.show()

#how many time steps/data pts are in one batch of data
seq_length=20

#generate evenly spaced data pts
time_steps = np.linspace(0,np.pi,seq_length+1)#等差函数序列，起始点，结束点，元素个数
data=np.sin(time_steps)
#print(data)
#print(type(data))
#print(data[2])
data.resize((seq_length+1,1))#size becomes(seq_length+1,1),adds an input_size dimention变成一个seq*1的二维数组（二层数组）,input  size就是1
#print(data)
print(type(data))

x=data[:-1]#all but the last piece of data,len x =20
#print(len(x))
y=data[1:]#all but the first

#display the data
plt.plot(time_steps[1:], x , 'r.', label='input,x')#x 参数意义：自变量取值，因变量，颜色，标签
plt.plot(time_steps[1:], y , 'b.', label='input,y')#y

plt.legend(loc='best')#图例
#plt.show()
class RNN(nn.Module):#继承nn,Module类。
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()#super是用来解决多重继承问题的，避免直接使用父类的名字。调用基类的__init__函数

        self.hidden_dim=hidden_dim

        #define an RNN with specified parameters
        #batch_first means that the first dim of the input and output will be the batch_
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        #last,fully-connected layer
        self.fc=nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):#hidden是hidden state
        #x(batch_size, seq_length, input_size)
        #hidden(n_layers, batch_size, input_size)
        #r_out(batch_size, time_step, hidden_size)
        batch_size=x.size(0)#获取tensor x的第一维个数。比如tensor([1,2,3]), size就是3

        #get RNN outputs
        r_out,hidden=self.rnn(x, hidden)
        #shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)#-1表示一个不确定的数字。

        #get final output
        output = self.fc(r_out)

        return output,hidden
'''
#test that dimensions are as expected
test_rnn=RNN(input_size=1, output_size=1,hidden_dim=10, n_layers=2)

test_input = torch.Tensor(data).unsqueeze(0)#give it a batch_size of 1 as first dimention
print('Input size:',test_input.size())#（1，20，1）batch_size,seq_len,input_size(input number of features)

#test out rnn sizes
test_out,test_h=test_rnn(test_input, None)#初始hidden state是没有
print('Output size:',test_out.size())#(20,1),seq_len,output_size
print('Hidden state size:',test_h.size())#(2,1,10)#number_size,batch_size.hidden_dim
'''
#decide on hyperparameters
input_size=1
output_size=1
hidden_dim=32
n_layers=1

#instantiate an RNN
rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)

#MSE loss and Adam optimizer with a learning rate of 0.01
criterion = nn.MSELoss()
print(criterion)
print(rnn.parameters())
optimizer = torch.optim.Adam(rnn.parameters(),lr=0.01)
#help(rnn)

#train the RNN
def train(rnn, n_steps, print_every):

    #initiatialize the hidden state
    hidden = None

    for batch_i, step in enumerate(range(n_steps)):#enumerate将一个可遍历的数据对象作为一个索引序列，同时列出数据和数据下标。
        #defining the training data
        time_steps = np.linspace(step * np.pi, (step+1)*np.pi, seq_length + 1)
        data = np.sin(time_steps)
        data.resize((seq_length + 1, 1))#input_size=1

        x=data[:-1]
        y=data[1:]

        #convert data into Tensors
        x_tensor = torch.Tensor(x).unsqueeze(0)#unsqueeze give a 1, （batch_size） dimension在第0维上加一维向量
        y_tensor = torch.Tensor(y)

        #outputs from the rnn
        prediction, hidden = rnn.forward(x_tensor, hidden)

        ##Representing Memory##
        #make a new varibale for hidden and detach the hidden state from its history
        #this way, we don't backpropagate through the entire history
        hidden = hidden.data

        #calculate the loss
        loss = criterion(prediction, y_tensor)#MSELoss(output_data, target)
        #zero gradients
        optimizer.zero_grad()#torch.optim.Adam(rnn.parameters(),lr=0.01)
        #perform backprop and update weights
        loss.backward()
        optimizer.step()

        #display loss and predictions
        if (batch_i % print_every) ==0:
            print(type(loss.item()))
            print('Loss: ', loss.item())
            plt.plot(time_steps[1:],x,'r.')#input
            plt.plot(time_steps[1:],prediction.data.numpy(),'b')#predictions
            plt.show()
    return rnn

#train the rnn and monitor results
n_steps = 75
print_every = 15

trained_rnn = train(rnn, n_steps, print_every)