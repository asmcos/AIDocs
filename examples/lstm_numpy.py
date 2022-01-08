# 这个让我疯狂的例子，就是来自下面地址
# https://github.com/AlphaDeeX/LSTM-Python
# 感谢作者，让我学到很多
# 在我对lstm好奇又无奈的时候，这段代码给了我希望
#
import numpy as np
import matplotlib.pyplot as plt
import sys



xrange = range

#
# keras standard_lstm 
#
'''
  def step(cell_inputs, cell_states):


    """Step function that will be used by Keras RNN backend."""
    h_tm1 = cell_states[0]  # previous memory state
    c_tm1 = cell_states[1]  # previous carry state

    z = backend.dot(cell_inputs, kernel)
    z += backend.dot(h_tm1, recurrent_kernel)
    z = backend.bias_add(z, bias)

    z0, z1, z2, z3 = tf.split(z, 4, axis=1)

    i = tf.sigmoid(z0)
    f = tf.sigmoid(z1)
    c = f * c_tm1 + i * tf.tanh(z2)
    o = tf.sigmoid(z3)

    h = o * tf.tanh(c)
    return h, [h, c]

'''
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

class LSTMPopulation(object):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.WLSTM = np.random.randn(1 + input_size + hidden_size, 4 * hidden_size) / np.sqrt(input_size + hidden_size)
        self.WLSTM[0,:] = 0
        self.WpeepIFO = np.ones((3, hidden_size))
        self.c0 = np.zeros((self.hidden_size))
        self.h0 = np.zeros((self.hidden_size))
        self.dcn = np.zeros((self.hidden_size))
        self.dhn = np.zeros((self.hidden_size))
        self.mWLSTM = np.zeros_like(self.WLSTM)
        self.mWpeepIFO = np.zeros_like(self.WpeepIFO)

    def reset_states(self):
        self.c0 = np.zeros((self.hidden_size))
        self.h0 = np.zeros((self.hidden_size))
        self.dcn = np.zeros((self.hidden_size))
        self.dhn = np.zeros((self.hidden_size))
        print ("Network states RESET")

    def forward(self, X):
        """
        X is the input (n,input_size). n = length of sequence, and input_size = input size (input dimension)
        """
        n = X.shape[0]

        # LSTM forward pass begins here
        xphpbias = self.WLSTM.shape[0] # x (input) plus, hidden plus, bias
        self.Hin = np.zeros((n, xphpbias))
        self.Hout = np.zeros((n, self.hidden_size))


        # z0,z1,z2,z3
        self.IFOA = np.zeros((n, self.hidden_size * 4)) #before non-linearlity
        # i,f,c,o
        self.IFOA_f = np.zeros((n, self.hidden_size * 4)) # after the non-linearity
        self.C = np.zeros((n, self.hidden_size)) # Cell values/ cell contents
        
        for t in xrange(n):
            prev_h = self.Hout[t-1,:] if (t > 0) else self.h0
            prev_c = self.C[t-1,:] if (t>0) else self.c0
            self.Hin[t,0] = 1 # this is for the bias
            self.Hin[t,1:1+self.input_size] = X[t, :]
            self.Hin[t,1+self.input_size:] = prev_h
            # Computing all gate activations 

            self.IFOA[t,:] = self.Hin[t,:].dot(self.WLSTM)
            z0,z1,z2,z3 = self.IFOA[t,:self.hidden_size],self.IFOA[t,self.hidden_size:2*self.hidden_size],\
                        self.IFOA[t,self.hidden_size*2:3*self.hidden_size],\
                        self.IFOA[t,self.hidden_size*3:]      
    
            # Adding peephole weights connections
            # self.IFOA[t,:self.hidden_size] = z0 +  np.multiply(prev_c, self.WpeepIFO[0,:])
            z0 = z0 +  np.multiply(prev_c, self.WpeepIFO[0,:])

            

            #self.IFOA[t,self.hidden_size:2*self.hidden_size] = z1 + np.multiply(prev_c, self.WpeepIFO[1,:])       # forget gate - adding peephole connections

            z1 = z1  + np.multiply(prev_c, self.WpeepIFO[1,:])


            # Passing through the non-linearities - sigmoid for gates input and forget - output is below due to peephole connections 
            #self.IFOA_f[t,0:2*self.hidden_size] = sigmoid(self.IFOA[t,0:2*self.hidden_size])

            i = sigmoid(z0)
            f = sigmoid(z1)
        
            # keras z2 = self.IFOA_f 3
            #self.IFOA_f[t,3*self.hidden_size:] = np.tanh(self.IFOA[t,3*self.hidden_size:]) # tanh non-linearity for the A gate (before the multiplicated input to the cell)
            c = np.tanh(z3)

            # Computing the cell activation            
            #self.C[t,:] = self.IFOA_f[t,self.hidden_size:2*self.hidden_size]*prev_c + self.IFOA_f[t,:self.hidden_size]*self.IFOA_f[t,3*self.hidden_size:]
            self.C[t,:] = f * prev_c + i * c

            # Computing the output gate with peephole connections
            #self.IFOA[t,2*self.hidden_size:3*self.hidden_size] = self.IFOA[t,2*self.hidden_size:3*self.hidden_size] + np.multiply(self.C[t,:], self.WpeepIFO[2,:]) # output gate - adding peephole connections            
            z2 = z2 +  np.multiply(self.C[t,:], self.WpeepIFO[2,:])
            
            #self.IFOA_f[t,2*self.hidden_size:3*self.hidden_size] = sigmoid(self.IFOA[t,2*self.hidden_size:3*self.hidden_size])
            o = sigmoid(z2)

            #self.Hout[t,:] = self.IFOA_f[t,2*self.hidden_size:3*self.hidden_size]*np.tanh(self.C[t,:])
            self.Hout[t,:] = o * np.tanh(self.C[t,:])

            self.IFOA[t,:self.hidden_size] = z0
            self.IFOA[t,self.hidden_size:2*self.hidden_size] = z1
            self.IFOA[t,2*self.hidden_size:3*self.hidden_size] = z2
            self.IFOA[t,3*self.hidden_size:] = z3

            self.IFOA_f[t,:self.hidden_size] = i
            self.IFOA_f[t,self.hidden_size:2*self.hidden_size] = f
            self.IFOA_f[t,2*self.hidden_size:3*self.hidden_size] = o
            self.IFOA_f[t,3*self.hidden_size:] = c

        self.c0 = self.C[t,:]
        self.h0 = self.Hout[t,:]

    def backward(self, dHout_temp):
        # backprop through the LSTM now
        self.dIFOA = np.zeros_like(self.IFOA)
        self.dIFOA_f = np.zeros_like(self.IFOA_f)
        self.dWLSTM = np.zeros_like(self.WLSTM)
        self.dWpeepIFO = np.zeros_like(self.WpeepIFO)
        self.dC = np.zeros_like(self.C)
        self.dHout = dHout_temp.copy()
        self.dHin = np.zeros_like(self.Hin)
        self.dh0 = np.zeros((self.hidden_size))

        n = self.Hin.shape[0]

        if self.dcn is not None: self.dC[n-1] += self.dcn.copy()
        if self.dhn is not None: self.dHout[n-1] += self.dhn.copy()

#        print(dHout.shape, C.shape)
        for t in reversed(xrange(n)):
            self.dIFOA_f[t,2*self.hidden_size:3*self.hidden_size] = self.dHout[t,:]*np.tanh(self.C[t,:]) # backprop in to output gate
            # backprop through the tanh non-linearity to get in to the cell, then will continue through it
            self.dC[t,:] += (self.dHout[t,:] * self.IFOA_f[t,2*self.hidden_size:3*self.hidden_size]) * (1 - np.tanh(self.C[t,:]**2))

            if (t>0):
                self.dIFOA_f[t,self.hidden_size:2*self.hidden_size] = self.dC[t,:]*self.C[t-1,:] # backprop in to the forget gate
                self.dC[t-1,:] += self.IFOA_f[t,self.hidden_size:2*self.hidden_size] * self.dC[t,:] # backprop through time for C (The recurrent connection to C from itself)
            else:
                self.dIFOA_f[t,self.hidden_size:2*self.hidden_size] = self.dC[t,:]*self.c0 # backprop in to forget gate
                self.dc0 = self.IFOA_f[t,self.hidden_size:2*self.hidden_size] * self.dC[t,:]

            self.dIFOA_f[t,:self.hidden_size] = self.dC[t,:]*self.IFOA_f[t,3*self.hidden_size:] #backprop in to the input gate
            self.dIFOA_f[t,3*self.hidden_size:] = self.dC[t,:]*self.IFOA_f[t,:self.hidden_size] #backprop in to the a gate                    

            # backprop through the activation functions
            # for input, forget and output gates - derivative of the sigmoid function
            # for a - derivative of the tanh function                

            self.dIFOA[t,3*self.hidden_size:] =  self.dIFOA_f[t,3*self.hidden_size:] * (1 - self.IFOA_f[t,3*self.hidden_size:]**2)
            y = self.IFOA_f[t,:3*self.hidden_size]
            self.dIFOA[t,:3*self.hidden_size] = (y*(1-y)) * self.dIFOA_f[t,:3*self.hidden_size]
    
            # backprop the input matrix multiplication            
            self.dWLSTM += np.dot(self.Hin[t:t+1,:].T, self.dIFOA[t:t+1,:])
            self.dHin[t,:] = self.dIFOA[t,:].dot(self.WLSTM.T)
            # backprop the peephole connections
            if t>0:
                self.dWpeepIFO[0,:] += np.multiply(self.dIFOA[t,:self.hidden_size], self.C[t-1,:])
                self.dWpeepIFO[1,:] += np.multiply(self.dIFOA[t,self.hidden_size:2*self.hidden_size], self.C[t-1,:])
                self.dWpeepIFO[2,:] += np.multiply(self.dIFOA[t,2*self.hidden_size:3*self.hidden_size], self.C[t,:])
            else:
                self.dWpeepIFO[0,:] += np.multiply(self.dIFOA[t,:self.hidden_size], self.c0)
                self.dWpeepIFO[1,:] += np.multiply(self.dIFOA[t,self.hidden_size:2*self.hidden_size], self.c0)
                self.dWpeepIFO[2,:] += np.multiply(self.dIFOA[t,2*self.hidden_size:3*self.hidden_size], self.C[t,:])

            if (t>0):
                self.dHout[t-1,:] += self.dHin[t,1+self.input_size:]
            else:
                self.dh0 += self.dHin[t,1+self.input_size:]

    def get_hidden_output(self):
        return self.Hout


    def train_network(self, learning_rate):
        for param, dparam, mem in zip([self.WLSTM, self.WpeepIFO],
                                  [self.dWLSTM, self.dWpeepIFO ],
                                  [self.mWLSTM, self.mWpeepIFO]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    def sample_network(self, X, W_out, next_data_points):
        self.reset_states()
        n = X.shape[0]
   
        p = n + next_data_points

        # LSTM forward pass for the duration of X, and then prediction for another n duration
        xphpbias = self.WLSTM.shape[0] # x (input) plus, hidden plus, bias
        self.Hin = np.zeros((p, xphpbias))
        self.Hout = np.zeros((p, self.hidden_size))

        self.IFOA = np.zeros((p, self.hidden_size * 4)) #before non-linearlity
        self.IFOA_f = np.zeros((p, self.hidden_size * 4)) # after the non-linearity
        self.C = np.zeros((p, self.hidden_size)) # Cell values/ cell contents

        for t in xrange(p):
            prev_h = self.Hout[t-1,:] if (t > 0) else self.h0
            prev_c = self.C[t-1,:] if (t>0) else self.c0

            self.Hin[t,0] = 1 # this is for the bias

            self.Hin[t,1+self.input_size:] = prev_h
            if (t<n):
                self.Hin[t,1:1+self.input_size] = X[t, :]
            else:
                self.Hin[t,1:1+self.input_size] = (self.Hout[t-1,:].dot(W_out.T))[0]
            # Computing all gate activations 
            self.IFOA[t,:] = self.Hin[t,:].dot(self.WLSTM)

            # Adding peephole weights connections
            self.IFOA[t,:self.hidden_size] = self.IFOA[t,:self.hidden_size] + np.multiply(prev_c, self.WpeepIFO[0,:])       # input gate - adding peephole connections
            self.IFOA[t,self.hidden_size:2*self.hidden_size] = self.IFOA[t,self.hidden_size:2*self.hidden_size] + np.multiply(prev_c, self.WpeepIFO[1,:])       # forget gate - adding peephole connections

            # Passing through the non-linearities - sigmoid for gates input and forget - output is below due to peephole connections 
            self.IFOA_f[t,0:2*self.hidden_size] = 1.0 / (1.0 + np.exp(-self.IFOA[t,0:2*self.hidden_size]))
            self.IFOA_f[t,3*self.hidden_size:] = np.tanh(self.IFOA[t,3*self.hidden_size:]) # tanh non-linearity for the A gate (before the multiplicated input to the cell)

            # Computing the cell activation            
            self.C[t,:] = self.IFOA_f[t,self.hidden_size:2*self.hidden_size]*prev_c + self.IFOA_f[t,:self.hidden_size]*self.IFOA_f[t,3*self.hidden_size:]

            # Computing the output gate with peephole connections
            self.IFOA[t,2*self.hidden_size:3*self.hidden_size] = self.IFOA[t,2*self.hidden_size:3*self.hidden_size] + np.multiply(self.C[t,:], self.WpeepIFO[2,:]) # output gate - adding peephole connections            

            self.IFOA_f[t,2*self.hidden_size:3*self.hidden_size] = 1.0 / (1.0 + np.exp(-self.IFOA[t,2*self.hidden_size:3*self.hidden_size]))
            self.Hout[t,:] = self.IFOA_f[t,2*self.hidden_size:3*self.hidden_size]*np.tanh(self.C[t,:])

        return self.Hout




def normalise(signal):
    mu = np.mean(signal)
    variance = np.mean((signal - mu)**2)
    signal_normalised = (signal - mu)/(np.sqrt(variance + 1e-8))
    return signal_normalised
    
t_range = np.linspace(0,100,1000)
train_df_roc_signal_unnormalised = np.sin(2*np.pi*300*t_range) + 0.5*np.sin(2*np.pi*t_range)

temp = train_df_roc_signal_unnormalised - min(train_df_roc_signal_unnormalised)
train_df_roc_signal = (temp)/max(temp)
plt.figure(1)
plt.plot(train_df_roc_signal[0:100])

seq_len = 24*3
input_size = 1
hidden_size_a = 200
output_size = 1
learning_rate = 1e-3
n, p = 0, 0

W_out = np.random.randn(output_size, hidden_size_a) / np.sqrt(output_size)
lstm_a = LSTMPopulation(input_size, hidden_size_a)
signal = np.zeros((seq_len,1))
target = np.zeros((seq_len,output_size))
mW_out = np.zeros_like(W_out)

j=0
k=0
for i in xrange(1000):
    # 将1000个数据生成步长是 24*3的结构，
    if j+seq_len+output_size >= len(train_df_roc_signal):
        j=0
        lstm_a.reset_states()
    signal[:,0] = train_df_roc_signal[j:j+seq_len]
    target[:,0] = train_df_roc_signal[j+1:j+1+seq_len]
    # 前馈
    lstm_a.forward(signal)
    lstm_a_hidden_out = lstm_a.get_hidden_output()  
    output = lstm_a_hidden_out.dot(W_out.T)   
    error = output - target
    dW_out = (error).T.dot(lstm_a_hidden_out)  
    # loss 是向后错一个数据的 24*3的结构
    loss = np.mean(np.square(output - target))  
    # error 是target
    dh_out = (error).dot(W_out)
    # 拿target数据 backward 后反馈
    lstm_a.backward(dh_out)
    lstm_a.train_network(learning_rate)        

    """
    for param, dparam, mem in zip([W_out],
                              [dW_out],
                              [mW_out]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
    """
    print (k, loss)

    k += 1
    j += 1

# Testing phase
for ll in range(1):
    index = 400+ll*100
    plot_len = 24*10
    next_vals = 12
    
    sample_signal = np.zeros((plot_len,1))
    sample_signal[:,0] = train_df_roc_signal[index:index+plot_len] 
    sample_signal_plotting = train_df_roc_signal[index:index + plot_len + next_vals] 
    
    dd = lstm_a.sample_network(sample_signal, W_out, next_vals)
    sampled_output = dd.dot(W_out.T)
    #y_out = dd.dot(W_out.T)
    #sampled_output = 1.0 / (1.0 + np.exp(-y_out))
    
    plt.figure(2)
    plt.plot(sampled_output[:,0])
    #plt.hold(True)
    plt.plot(sample_signal_plotting[:], 'r')
    plt.title('Prediction vs Actual Signal')
    
    plt.figure(3)
    plt.plot(sampled_output[plot_len:plot_len+next_vals,0])
    #plt.hold(True)
    plt.plot(sample_signal_plotting[plot_len:plot_len+next_vals], 'r')
    plt.title('Prediction Mode - Blue (Prediction), Red (Actual)')
    #plt.hold(False)
    plt.show()
