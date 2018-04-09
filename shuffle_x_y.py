import numpy as np
x=np.load('x.npy')
y=np.load('y.npy')
y=y.argmax(axis=1)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x = x[shuffle_indices]
y = y[shuffle_indices]
train_len = int(len(x) * 0.9)
x_train = x[:train_len]
y_train = y[:train_len]
x_test = x[train_len:]
y_test = y[train_len:]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
np.save('x_trian.npy',x_train)
np.save('x_test.npy',x_test)
np.save('y_train.npy',y_train)
np.save('y_test.npy',y_test)