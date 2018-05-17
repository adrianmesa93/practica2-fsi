import gzip
import pickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
train_set, valid_set, test_set = u.load()
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

import matplotlib.cm as cm
import matplotlib.pyplot as plt

train_y = one_hot(train_y, 10)  # the labels are in the last row. Then we encode them in one hot code
test_y2 = one_hot(test_y, 10)  # the labels are in the last row. Then we encode them in one hot code
x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

# Salida de las neuronas

W2 = tf.Variable(np.float32(np.random.rand(5, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)  # Cálculo sigmoide
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)  #

loss = tf.reduce_sum(tf.square(y_ - y)) #error

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.05
#train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print ("----------------------")
print ("   Start training...  ")
print ("----------------------")

batch_size = 20

training1 = [];
validation1 = [];
epoch=0;
estabilidad=0
while estabilidad <15:
    for jj in range(int(len(train_x) // batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    training2 = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})/batch_size
    validation2 = sess.run(loss, feed_dict={x: valid_x, y_: one_hot(valid_y,10)})/len(test_y2)
    validation1.append(validation2)
    training1.append(training2)

    print ("Epoch #:", epoch, "Error: ", training2)
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print (b, "-->", r)
    print ("----------------------------------------------------------------------------------")
    epoch = epoch +1
    print ("Estabilidad: ",estabilidad)
    if (len(validation1) >= 2 and abs(validation1[epoch-2] - validation1[epoch-1]) < 0.05):
        estabilidad+=1
    else:
        estabilidad=0
# Test
print ("Resultados del test:")
print ("----------------------------------------------------------------------------------")
result=sess.run(y,feed_dict={x: test_x})
success = 0
fail = 0
for b, r in zip(test_y, result):
    if (np.argmax(b) == np.argmax(r)):
        success += 1
    else:
        fail += 1
    #print b, "-->", r
total = success + fail
print ("Numero de aciertos: " , success)
print ("Numero de fallos: " , fail)
print ("Numero total: " , total)
print ("Porcentaje de aciertos: " , (float(success) / float(total)) * 100 , "%")
print ("----------------------------------------------------------------------------------")

x_axis_training = list(range(1, len(training1)+1))
plt.plot(x_axis_training, validation1, training1)
plt.show()
# ---------------- Visualizing some element of the MNIST dataset --------------

#
#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print (train_y[57])


# TODO: the neural net!!
