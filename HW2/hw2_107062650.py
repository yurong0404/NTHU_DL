from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

df = pd.read_csv('Data.csv')
#divide into train and test dataframe
train_df = df.sample(frac=0.8)
test_df = df.drop(train_df.index)
train_df = train_df.reset_index()
test_df = test_df.reset_index()
test_type = test_df['Activities_Types']
train_y = pd.get_dummies(train_df['Activities_Types'])
test_y = pd.get_dummies(test_df['Activities_Types'])
train_df = train_df.drop(columns=['index', 'Activities_Types'])
test_df = test_df.drop(columns=['index', 'Activities_Types'])

#normalize
train_df = (train_df-train_df.mean())/train_df.std()
test_df = (test_df-test_df.mean())/test_df.std()
train_x = train_df.values
test_x = test_df.values
train_y =train_y.values
test_y =test_y.values

#read hidden test data
hidden_df = pd.read_csv('Test_no_Ac.csv')

# Parameters
learning_rate = 0.01
num_steps = 150
batch_size = 1500
display_step = 10
# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 68
num_classes = 6



X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {
	'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'out': tf.Variable(tf.random_normal([num_classes]))
}



def get_Batch(data, label, batch_size):
    input_queue = tf.train.slice_input_producer([data, label], capacity=64, shuffle=True) 
    x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=64, allow_smaller_final_batch=True)
    return x_batch, y_batch

# Create model
def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
gd_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op_gd = gd_optimizer.minimize(loss_op)
ada_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train_op_ada = ada_optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
x_batch, y_batch = get_Batch(train_x, train_y, batch_size)


lossArray = np.array([])
lossValidateArr = np.array([])
accArray = np.array([])
accValidateArr = np.array([])

def countFactor(predict, actual, classType):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i in range(len(predict)):
        preType = np.argmax(predict[i])+1
        actType = np.argmax(actual[i])+1
        if preType == classType and actType == classType:
            TP += 1
        elif preType != classType and (preType==actType):
            TN += 1
        elif preType == classType and actType!=classType:
            FP += 1
        elif preType != classType and (preType!=actType):
            FN += 1
    return TP,TN,FP,FN

def countMetric(TP, TN, FP, FN):
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f_score = 2*(precision*recall)/(precision+recall)
    return precision, recall, f_score

def countMacAvg(p1,p2,p3,p4,p5,p6,r1,r2,r3,r4,r5,r6,f1,f2,f3,f4,f5,f6):
    macP = (p1+p2+p3+p4+p5+p6)/6
    macR = (r1+r2+r3+r4+r5+r6)/6
    macF = 2*(macP*macR)/(macP+macR)
    return macP,macR,macF

def countMicAvg(TP1,TP2,TP3,TP4,TP5,TP6,FP1,FP2,FP3,FP4,FP5,FP6,FN1,FN2,FN3,FN4,FN5,FN6):
    TP_avg = (TP1+TP2+TP3+TP4+TP5+TP6)/6
    FP_avg = (FP1+FP2+FP3+FP4+FP5+FP6)/6
    FN_avg = (FN1+FN2+FN3+FN4+FN5+FN6)/6
    micP = TP_avg/(TP_avg+FP_avg)
    micR = TP_avg/(TP_avg+FN_avg)
    micF = 2*(micP*micR)/(micP+micR)
    return micP,micR,micF

# Start training
with tf.Session() as sess:
# Run the initializer
    print('Adam Optimizer start')
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    for step in range(1, num_steps+1):
        if coord.should_stop():
            break
        batch_x, batch_y = sess.run([x_batch, y_batch])
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,Y: batch_y})
        lossArray = np.append(lossArray, loss)
        accArray = np.append(accArray, acc)
        
        lossValid, accValidate = sess.run([loss_op, accuracy], feed_dict={X: test_x, Y: test_y})
        lossValidateArr = np.append(lossValidateArr, lossValid)
        accValidateArr = np.append(accValidateArr, accValidate)
        if step % display_step == 0 or step == 1:
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    coord.request_stop()
    coord.join(threads)
    print("Optimization Finished!")

    plt.plot(range(1, num_steps+1), lossArray, "b", label="train")
    plt.plot(range(1, num_steps+1), lossValidateArr, "g", label="validate")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Adam Optimizer model loss")
    plt.legend(loc = 2)
    plt.show()
    
    plt.plot(range(1, num_steps+1), accArray, "b", label="train")
    plt.plot(range(1, num_steps+1), accValidateArr, "g", label="validate")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Adam Optimizer model accuracy")
    plt.legend(loc = 2)
    plt.show()
    print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={X: test_x,Y: test_y}))
    predictRes = sess.run(prediction, feed_dict={X: test_x,Y: test_y})
    
    hidden_predict_onehot = sess.run(prediction, feed_dict={X: hidden_df.values})
    hidden_predict = np.argmax(hidden_predict_onehot, axis=1)+1
    f = open("107062650_answer.txt", "w")
    for index, item in enumerate(hidden_predict):
        f.write(str(index+1)+'\t')
        f.write(str(item))
        f.write('\n')
    f.close()
    print('Hidden test answer have been writen.')
    
    TP1, TN1, FP1, FN1 = countFactor(predictRes, test_y, 1)
    TP2, TN2, FP2, FN2 = countFactor(predictRes, test_y, 2)
    TP3, TN3, FP3, FN3 = countFactor(predictRes, test_y, 3)
    TP4, TN4, FP4, FN4 = countFactor(predictRes, test_y, 4)
    TP5, TN5, FP5, FN5 = countFactor(predictRes, test_y, 5)
    TP6, TN6, FP6, FN6 = countFactor(predictRes, test_y, 6)
    precision1, recall1, f_score1 = countMetric(TP1,TN1,FP1,FN1)
    precision2, recall2, f_score2 = countMetric(TP2,TN2,FP2,FN2)
    precision3, recall3, f_score3 = countMetric(TP3,TN3,FP3,FN1)
    precision4, recall4, f_score4 = countMetric(TP4,TN4,FP4,FN4)
    precision5, recall5, f_score5 = countMetric(TP5,TN5,FP5,FN5)
    precision6, recall6, f_score6 = countMetric(TP6,TN6,FP6,FN6)
    print('Class 1')
    print('precision: ',format(precision1, '0.3f'),'recall:',format(recall1, '0.3f'),'f-score:',format(f_score1,'0.3f'))
    print('Class 2')
    print('precision: ',format(precision2, '0.3f'),'recall:',format(recall2, '0.3f'),'f-score:',format(f_score2,'0.3f'))
    print('Class 3')
    print('precision: ',format(precision3, '0.3f'),'recall:',format(recall3, '0.3f'),'f-score:',format(f_score3,'0.3f'))
    print('Class 4')
    print('precision: ',format(precision4, '0.3f'),'recall:',format(recall4, '0.3f'),'f-score:',format(f_score4,'0.3f'))
    print('Class 5')
    print('precision: ',format(precision5, '0.3f'),'recall:',format(recall5, '0.3f'),'f-score:',format(f_score5,'0.3f'))
    print('Class 6')
    print('precision: ',format(precision6, '0.3f'),'recall:',format(recall6, '0.3f'),'f-score:',format(f_score6,'0.3f'))
    macP, macR, macF = countMacAvg(precision1, precision2, precision3, precision4, precision5, precision6, \
                                                        recall1, recall2, recall3, recall4, recall5, recall6,\
                                                        f_score1, f_score2, f_score3, f_score4, f_score5, f_score6)
    print('macro precision: ', format(macP, '0.3f'))
    print('macro recall: ', format(macR, '0.3f'))
    print('macro f-score: ', format(macF, '0.3f'))
    
    micP, micR, micF = countMicAvg(TP1,TP2,TP3,TP4,TP5,TP6,FP1,FP2,FP3,FP4,FP5,FP6,FN1,FN2,FN3,FN4,FN5,FN6)
    print('micro precision: ', format(micP, '0.3f'))
    print('micro recall', format(micR, '0.3f'))
    print('micro f-scroe', format(micF, '0.3f'))
    
with tf.Session() as sess:
    #try gradient descent optimizer
    lossArray = np.array([])
    lossValidateArr = np.array([])
    accArray = np.array([])
    accValidateArr = np.array([])
    print('\nGradient Descent Optimizer start')
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    for step in range(1, num_steps+1):
        if coord.should_stop():
            break
        batch_x, batch_y = sess.run([x_batch, y_batch])
        # Run optimization op (backprop)
        sess.run(train_op_gd, feed_dict={X: batch_x, Y: batch_y})
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,Y: batch_y})
        lossArray = np.append(lossArray, loss)
        accArray = np.append(accArray, acc)
        
        lossValid, accValidate = sess.run([loss_op, accuracy], feed_dict={X: test_x, Y: test_y})
        lossValidateArr = np.append(lossValidateArr, lossValid)
        accValidateArr = np.append(accValidateArr, accValidate)
        if step % display_step == 0 or step == 1:
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    coord.request_stop()
    coord.join(threads)
    print("Optimization Finished!")

    plt.plot(range(1, num_steps+1), lossArray, "b", label="train")
    plt.plot(range(1, num_steps+1), lossValidateArr, "g", label="validate")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Gradient Descent Optimizer model loss")
    plt.legend(loc = 2)
    plt.show()
    
    plt.plot(range(1, num_steps+1), accArray, "b", label="train")
    plt.plot(range(1, num_steps+1), accValidateArr, "g", label="validate")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Gradient Descent Optimizer model accuracy")
    plt.legend(loc = 2)
    plt.show()
    print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={X: test_x,Y: test_y}))
    predictRes = sess.run(prediction, feed_dict={X: test_x,Y: test_y})
    
    TP1, TN1, FP1, FN1 = countFactor(predictRes, test_y, 1)
    TP2, TN2, FP2, FN2 = countFactor(predictRes, test_y, 2)
    TP3, TN3, FP3, FN3 = countFactor(predictRes, test_y, 3)
    TP4, TN4, FP4, FN4 = countFactor(predictRes, test_y, 4)
    TP5, TN5, FP5, FN5 = countFactor(predictRes, test_y, 5)
    TP6, TN6, FP6, FN6 = countFactor(predictRes, test_y, 6)
    precision1, recall1, f_score1 = countMetric(TP1,TN1,FP1,FN1)
    precision2, recall2, f_score2 = countMetric(TP2,TN2,FP2,FN2)
    precision3, recall3, f_score3 = countMetric(TP3,TN3,FP3,FN1)
    precision4, recall4, f_score4 = countMetric(TP4,TN4,FP4,FN4)
    precision5, recall5, f_score5 = countMetric(TP5,TN5,FP5,FN5)
    precision6, recall6, f_score6 = countMetric(TP6,TN6,FP6,FN6)
    print('Class 1')
    print('precision: ',format(precision1, '0.3f'),'recall:',format(recall1, '0.3f'),'f-score:',format(f_score1,'0.3f'))
    print('Class 2')
    print('precision: ',format(precision2, '0.3f'),'recall:',format(recall2, '0.3f'),'f-score:',format(f_score2,'0.3f'))
    print('Class 3')
    print('precision: ',format(precision3, '0.3f'),'recall:',format(recall3, '0.3f'),'f-score:',format(f_score3,'0.3f'))
    print('Class 4')
    print('precision: ',format(precision4, '0.3f'),'recall:',format(recall4, '0.3f'),'f-score:',format(f_score4,'0.3f'))
    print('Class 5')
    print('precision: ',format(precision5, '0.3f'),'recall:',format(recall5, '0.3f'),'f-score:',format(f_score5,'0.3f'))
    print('Class 6')
    print('precision: ',format(precision6, '0.3f'),'recall:',format(recall6, '0.3f'),'f-score:',format(f_score6,'0.3f'))
    macP, macR, macF = countMacAvg(precision1, precision2, precision3, precision4, precision5, precision6, \
                                                        recall1, recall2, recall3, recall4, recall5, recall6,\
                                                        f_score1, f_score2, f_score3, f_score4, f_score5, f_score6)
    print('macro precision: ', format(macP, '0.3f'))
    print('macro recall: ', format(macR, '0.3f'))
    print('macro f-score: ', format(macF, '0.3f'))
    
    micP, micR, micF = countMicAvg(TP1,TP2,TP3,TP4,TP5,TP6,FP1,FP2,FP3,FP4,FP5,FP6,FN1,FN2,FN3,FN4,FN5,FN6)
    print('micro precision: ', format(micP, '0.3f'))
    print('micro recall', format(micR, '0.3f'))
    print('micro f-scroe', format(micF, '0.3f'))
    
with tf.Session() as sess:
    lossArray = np.array([])
    lossValidateArr = np.array([])
    accArray = np.array([])
    accValidateArr = np.array([])
    print('\nAda-grad Optimizer start')
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    for step in range(1, num_steps+1):
        if coord.should_stop():
            break
        batch_x, batch_y = sess.run([x_batch, y_batch])
        # Run optimization op (backprop)
        sess.run(train_op_ada, feed_dict={X: batch_x, Y: batch_y})
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,Y: batch_y})
        lossArray = np.append(lossArray, loss)
        accArray = np.append(accArray, acc)
        
        lossValid, accValidate = sess.run([loss_op, accuracy], feed_dict={X: test_x, Y: test_y})
        lossValidateArr = np.append(lossValidateArr, lossValid)
        accValidateArr = np.append(accValidateArr, accValidate)
        if step % display_step == 0 or step == 1:
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    coord.request_stop()
    coord.join(threads)
    print("Optimization Finished!")

    plt.plot(range(1, num_steps+1), lossArray, "b", label="train")
    plt.plot(range(1, num_steps+1), lossValidateArr, "g", label="validate")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Ada-grad Optimizer model loss")
    plt.legend(loc = 2)
    plt.show()
    
    plt.plot(range(1, num_steps+1), accArray, "b", label="train")
    plt.plot(range(1, num_steps+1), accValidateArr, "g", label="validate")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Ada-grad Optimizer model accuracy")
    plt.legend(loc = 2)
    plt.show()
    print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={X: test_x,Y: test_y}))
    predictRes = sess.run(prediction, feed_dict={X: test_x,Y: test_y})
    
    TP1, TN1, FP1, FN1 = countFactor(predictRes, test_y, 1)
    TP2, TN2, FP2, FN2 = countFactor(predictRes, test_y, 2)
    TP3, TN3, FP3, FN3 = countFactor(predictRes, test_y, 3)
    TP4, TN4, FP4, FN4 = countFactor(predictRes, test_y, 4)
    TP5, TN5, FP5, FN5 = countFactor(predictRes, test_y, 5)
    TP6, TN6, FP6, FN6 = countFactor(predictRes, test_y, 6)
    precision1, recall1, f_score1 = countMetric(TP1,TN1,FP1,FN1)
    precision2, recall2, f_score2 = countMetric(TP2,TN2,FP2,FN2)
    precision3, recall3, f_score3 = countMetric(TP3,TN3,FP3,FN1)
    precision4, recall4, f_score4 = countMetric(TP4,TN4,FP4,FN4)
    precision5, recall5, f_score5 = countMetric(TP5,TN5,FP5,FN5)
    precision6, recall6, f_score6 = countMetric(TP6,TN6,FP6,FN6)
    print('Class 1')
    print('precision: ',format(precision1, '0.3f'),'recall:',format(recall1, '0.3f'),'f-score:',format(f_score1,'0.3f'))
    print('Class 2')
    print('precision: ',format(precision2, '0.3f'),'recall:',format(recall2, '0.3f'),'f-score:',format(f_score2,'0.3f'))
    print('Class 3')
    print('precision: ',format(precision3, '0.3f'),'recall:',format(recall3, '0.3f'),'f-score:',format(f_score3,'0.3f'))
    print('Class 4')
    print('precision: ',format(precision4, '0.3f'),'recall:',format(recall4, '0.3f'),'f-score:',format(f_score4,'0.3f'))
    print('Class 5')
    print('precision: ',format(precision5, '0.3f'),'recall:',format(recall5, '0.3f'),'f-score:',format(f_score5,'0.3f'))
    print('Class 6')
    print('precision: ',format(precision6, '0.3f'),'recall:',format(recall6, '0.3f'),'f-score:',format(f_score6,'0.3f'))
    macP, macR, macF = countMacAvg(precision1, precision2, precision3, precision4, precision5, precision6, \
                                                        recall1, recall2, recall3, recall4, recall5, recall6,\
                                                        f_score1, f_score2, f_score3, f_score4, f_score5, f_score6)
    print('macro precision: ', format(macP, '0.3f'))
    print('macro recall: ', format(macR, '0.3f'))
    print('macro f-score: ', format(macF, '0.3f'))
    
    micP, micR, micF = countMicAvg(TP1,TP2,TP3,TP4,TP5,TP6,FP1,FP2,FP3,FP4,FP5,FP6,FN1,FN2,FN3,FN4,FN5,FN6)
    print('micro precision: ', format(micP, '0.3f'))
    print('micro recall', format(micR, '0.3f'))
    print('micro f-scroe', format(micF, '0.3f'))
    
sc = StandardScaler()
Z = sc.fit_transform(test_df)
R = np.dot(Z.T, Z) / df.shape[0]
eigen_vals, eigen_vecs = np.linalg.eigh(R)

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(reverse=True)

W = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))

Z_pca = Z.dot(W)

plt.figure(figsize=(12,8))
colors = ['r', 'b', 'g','c','m','y']
markers = ['o', 'o', 'o', 'o', 'o', 'o']
labels = ['dws','ups','sit','std','wlk','jog']
for l, c, lb, m in zip(np.unique(test_type.values), colors, labels, markers):
    plt.scatter(Z_pca[test_type.values==l, 0], 
                Z_pca[test_type.values==l, 1], 
                c=c, label=lb, marker=m)

plt.title('PCA')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

tsne2 = TSNE(n_components=2, random_state=0)
tsne2_result = tsne2.fit_transform(test_df)

plt.figure(figsize=(12,8))
colors = ['r', 'b', 'g','c','m','y']
markers = ['o', 'o', 'o', 'o', 'o', 'o']
labels = ['dws','ups','sit','std','wlk','jog']
for l, c, lb, m in zip(np.unique(test_type.values), colors, labels, markers):
    plt.scatter(tsne2_result[test_type.values==l, 0], 
                tsne2_result[test_type.values==l, 1], 
                c=c, label=lb, marker=m)

plt.title('t-SNE')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()