
from __future__ import print_function
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import csv

def loadCSVfile():
    tmp = np.loadtxt("data.csv", dtype=np.float, delimiter=",")
    x = tmp[0:140,1:].astype(np.float)
    y = tmp[0:140,0:1].astype(np.float)
    x_test = tmp[140:,1:].astype(np.float)
    y_test = tmp[140:,0:1].astype(np.float)
    return x,y,x_test,y_test

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


x,y,x_test,y_test = loadCSVfile()
rows,columes = x.shape

#feature scaling
min_max_scaler_x = preprocessing.MinMaxScaler().fit(x)
min_max_scaler_y = preprocessing.MinMaxScaler().fit(y)

x_data = min_max_scaler_x.fit_transform(x)
y_data = min_max_scaler_y.fit_transform(y)

x_data_test = min_max_scaler_x.fit_transform(x_test)
y_data_test = min_max_scaler_y.fit_transform(y_test)


# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, columes], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')   

# add hidden layer
l1_1 = add_layer(xs[:,0:10], 10, 20, n_layer='hidden1_1', activation_function=tf.nn.relu)
l1_2 = add_layer(xs[:,10:18], 7, 20, n_layer='hidden1_2', activation_function=tf.nn.relu)

l2_1 = add_layer(l1_1, 20, 5, n_layer='hidden2_1', activation_function=tf.nn.relu)
l2_2 = add_layer(l1_2, 20, 5, n_layer='hidden2_2', activation_function=tf.nn.relu)

# add output layer
prediction = add_layer(l2_1, 5, 1, n_layer='prediction1', activation_function=None)
prediction2 = add_layer(l2_2, 5, 1, n_layer='prediction2', activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - (prediction)),
                                        reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

with tf.name_scope('acc'):
    acc = tf.reduce_mean(tf.reduce_sum(tf.square(ys - (prediction-prediction2)),
                                        reduction_indices=[1]))
    tf.summary.scalar('acc', acc)

sess = tf.Session()
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("logs/", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(100000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result,acc = sess.run([merged,test],
                          feed_dict={xs: x_data, ys: y_data})
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        # print(sess.run(tf.reduce_mean(tf.reduce_sum(prediction,reduction_indices=[1])),feed_dict={xs:x_data,ys:y_data}))
        # print(sess.run(tf.reduce_mean(tf.reduce_sum(prediction,reduction_indices=[1])),feed_dict={xs:x_data,ys:y_data}))
        # print(sess.run(tf.reduce_sum(prediction2,reduction_indices=[1]),feed_dict={xs:x_data,ys:y_data}))
        # print(sess.run(tf.reduce_mean(tf.reduce_sum(tf.square(ys - (prediction-prediction2)),
        #                                 reduction_indices=[1])), feed_dict={xs: x_data_test, ys: y_data_test}))
        writer.add_summary(result, i)

a = []
b = []
c = []
d = []

E = 4
R = 7
C = 8
N = 164
points = [0.1,0.3,0.5,0.7,0.9]
for i in points:
    for j in points:
        for k in points:
            x_data[N][C] = i
            x_data[N][R] = j
            x_data[N][E] = k
            #print(min_max_scaler.inverse_transform(sess.run(ys-(prediction-prediction2),feed_dict={xs: x_data, ys: y_data})))
            #a = min_max_scaler_y.inverse_transform(sess.run(tf.reduce_mean(tf.square(ys-(prediction-prediction2))),feed_dict={xs: x_data, ys: y_data}))
            a.append(sess.run(tf.reduce_mean(prediction[N]-prediction2[N]),feed_dict={xs: x_data, ys: y_data}))
            b.append(x_data[N][C]) #全球平均值
            c.append(sess.run(tf.reduce_mean(prediction[N]),feed_dict={xs: x_data, ys: y_data}))
            d.append(sess.run(tf.reduce_mean(prediction2[N]),feed_dict={xs: x_data, ys: y_data}))
            print(i,j,k)
#print(a,b,c,d)
mse = sess.run(tf.reduce_mean(tf.square(ys-(prediction-prediction2))),,feed_dict={xs: x_data, ys: y_data})
#print(mse)