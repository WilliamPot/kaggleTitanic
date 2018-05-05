# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:07:38 2018

@author: Chen
"""
from imTools import csvP
import pandas as pds
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
ops.reset_default_graph()
train_data = pds.read_csv('train.csv')
beiyong_data = pds.read_csv('train.csv')
test_data = pds.read_csv('test.csv')
#丢弃无关列
train_data = train_data.drop(['Cabin','Ticket','PassengerId'],axis= 1)
test_data = test_data.drop(['Cabin','Ticket','PassengerId'],axis= 1)
#标注性别,登船地
male_index= train_data[train_data['Sex']=='male'].index
female_index= train_data[train_data['Sex']=='female'].index
train_data.ix[male_index,'Sex']= 1
train_data.ix[female_index,'Sex']= 2     
male_test_index= test_data[test_data['Sex']=='male'].index
female_test_index= test_data[test_data['Sex']=='female'].index
test_data.ix[male_test_index,'Sex']= 1
test_data.ix[female_test_index,'Sex']= 2 
C_index= train_data[train_data['Embarked']=='C'].index
Q_index= train_data[train_data['Embarked']=='Q'].index
S_index= train_data[train_data['Embarked']=='S'].index
train_data.ix[C_index,'Embarked']= 1
train_data.ix[Q_index,'Embarked']= 2  
train_data.ix[S_index,'Embarked']= 3  
C_index= test_data[test_data['Embarked']=='C'].index
Q_index= test_data[test_data['Embarked']=='Q'].index
S_index= test_data[test_data['Embarked']=='S'].index
test_data.ix[C_index,'Embarked']= 1
test_data.ix[Q_index,'Embarked']= 2  
test_data.ix[S_index,'Embarked']= 3 
index= train_data[train_data['Embarked'].isnull()].index
nums = len(index)
train_data.ix[index,'Embarked'] = np.random.choice(4,nums)
#index20= train_data[train_data['Age']<=20]
#index50= train_data[train_data['Age']>=50]
#train_data = train_data.append(index20,ignore_index = True)
#train_data = train_data.append(index50,ignore_index = True)
#Fare归一化

def normalize_fare(df):
    fares = df['Fare'].values
    min_fare = min(fares)
    max_fare = max(fares)
    normalized_fares = []
    for fare in fares:
        n_fare = (fare-min_fare)/(max_fare-min_fare)
        normalized_fares.append(n_fare)
    df.ix[:,'Fare'] = normalized_fares
normalize_fare(train_data)
normalize_fare(test_data)

#预测年龄
train_index= train_data[train_data['Age'].notnull()].index
age_pre_train_data = np.array(train_data.ix[train_index,['Pclass','Parch','SibSp','Sex','Embarked']].values).astype(np.float64)
age_pre_train_label = train_data.ix[train_index,['Age']].values
test_index = test_data[test_data['Age'].notnull()].index
age_pre_test_data = np.array(test_data.ix[test_index,['Pclass','Parch','SibSp','Sex','Embarked']].values).astype(np.float64)
age_pre_test_label = test_data.ix[test_index,['Age']].values  
                                 
batch_size = 32
with tf.name_scope('inputs'):
    x_data= tf.placeholder(shape = [None,5],dtype = tf.float32,name='x_input')
    y_target = tf.placeholder(shape = [None,1],dtype = tf.float32,name='y_input')
hidden_layer_nodes = 8
hidden_layer_nodes2 = 12
with tf.name_scope('layer1'):
    with tf.name_scope('weights1'):
        A1 = tf.Variable(tf.random_normal(shape=[5,hidden_layer_nodes]))
        tf.summary.histogram('weights1',A1)
    with tf.name_scope('bias1'):
        b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
        tf.summary.histogram('bias1',b1)
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data,A1),b1))
with tf.name_scope('layer2'):
    with tf.name_scope('weight2'):
        A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,hidden_layer_nodes2]))
        tf.summary.histogram('weights2',A2)
    with tf.name_scope('bias2'):
        b2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes2]))
        tf.summary.histogram('bias2',b2)
hidden_output2 = tf.add(tf.matmul(hidden_output,A2),b2)
with tf.name_scope('layer3'):
    with tf.name_scope('weight3'):
        A3 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes2,1]))
        tf.summary.histogram('weights3',A3)
    with tf.name_scope('bias3'):
        b3 = tf.Variable(tf.random_normal(shape=[1]))
        tf.summary.histogram('bias3',b3)
final_output = tf.add(tf.matmul(hidden_output2,A3),b3)
loss= tf.reduce_mean(tf.square(y_target-final_output))
my_opt= tf.train.AdamOptimizer(0.002)
train_step = my_opt.minimize(loss)
init=tf.global_variables_initializer()
sess = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('kaggle',sess.graph)
sess.run(init)

loss_vec = []
test_loss = []
for i in range(20000):
    summary = sess.run(merged)
    train_writer.add_summary(summary, i+1)
    rand_index = np.random.choice(len(age_pre_train_data),replace=False,size = batch_size)
    rand_x = age_pre_train_data[rand_index]
    rand_y = age_pre_train_label[rand_index]
    rand_test_index = np.random.choice(len(age_pre_test_data),size = 200)
    sess.run(train_step,feed_dict = {x_data:rand_x,y_target:rand_y})
    temp_loss = sess.run(loss,feed_dict = {x_data:rand_x,y_target:rand_y})
    loss_vec.append(np.sqrt(temp_loss))
    test_temp_loss = sess.run(loss,feed_dict = {x_data:age_pre_test_data[rand_test_index],y_target:age_pre_test_label[rand_test_index]})
    test_loss.append(np.sqrt(test_temp_loss))
    if (i+1)%50 == 0:
        print('Generation: {}.Loss = {}'.format(i+1,temp_loss))
        
index= train_data[train_data['Age'].isnull()].index
notnull_index= train_data[train_data['Age'].notnull()].index
prediction_label = train_data.ix[notnull_index,'Age'].values
prediction_data= np.array(train_data.ix[notnull_index,['Pclass','Parch','SibSp','Sex','Embarked']].values).astype(np.float64)
prediction = sess.run(final_output,feed_dict = {x_data:prediction_data}) 

sup_data = np.array(train_data.ix[index,['Pclass','Parch','SibSp','Sex','Embarked']].values).astype(np.float64)
supplement = sess.run(final_output,feed_dict = {x_data:sup_data})
prob = np.mean(np.abs(prediction-prediction_label)<20)*100
print('Probability that the differences are within 20: {}%'.format(prob))
#结果差异在20岁以内的概率在77%左右,够了，填补确实数据
sup = supplement.astype(np.int32).astype(np.float64)
train_data.ix[index,'Age'] = sup
             
index= test_data[test_data['Age'].isnull()].index

sup_data = np.array(test_data.ix[index,['Pclass','Parch','SibSp','Sex','Embarked']].values).astype(np.float64)
supplement = sess.run(final_output,feed_dict = {x_data:sup_data})

sup = supplement.astype(np.int32).astype(np.float64)
test_data.ix[index,'Age'] = sup
            
                    
print('Supplementing data finished.')
print('Start training.')


def normalize_age(df):
    ages = df['Age'].values
    min_age = min(ages)
    max_age = max(ages)
    normalized_ages = []
    for age in ages:
        n_age = (age-min_age)/(max_age-min_age)
        normalized_ages.append(n_age)
    df.ix[:,'Age'] = normalized_ages
normalize_age(train_data)
normalize_age(test_data)
          
train = np.array(train_data.ix[:,['Pclass','Parch','SibSp','Sex','Age','Embarked']].values).astype(np.float64)
labels = np.array(train_data.ix[:,['Survived']].values).astype(np.int64)
labels = np.squeeze(labels,1)

data = tf.placeholder(shape = [None,6],dtype = tf.float32)
label = tf.placeholder(shape = [None],dtype = tf.int64)                           
A3 = tf.Variable(tf.random_normal(shape=[6,hidden_layer_nodes]))
b3 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
h_output = tf.nn.relu(tf.add(tf.matmul(data,A3),b3))
A4 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,hidden_layer_nodes2]))
b4 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes2]))
h_output2 = tf.add(tf.matmul(h_output,A4),b4)
A5 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes2,2]))
b5 = tf.Variable(tf.random_normal(shape=[2]))
f_output = tf.add(tf.matmul(h_output2,A5),b5) 
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=f_output, labels=label) # logits=float32, labels=int32
loss2 = tf.reduce_mean(losses)
tf.summary.scalar('loss',loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(f_output, 1), label), tf.float32))

optimizer = tf.train.AdamOptimizer(0.002)
train_step2 = optimizer.minimize(loss2)

init = tf.global_variables_initializer()
sess.run(init)

batch_size1 = 100
steps = []
loss= []
print('Start training.')
for i in range(20000):
    rand_index = np.random.choice(len(train),replace=False,size = batch_size1)
    randx = train[rand_index]
    randy = labels[rand_index]
    sess.run(train_step2,feed_dict = {data:randx,label:randy})
    temp_loss = sess.run(loss2,feed_dict = {data:randx,label:randy})
    if (i+1)%50 == 0:
        loss.append(temp_loss)
        steps.append(i+1)
        acc = sess.run(accuracy,feed_dict = {data:randx,label:randy})
        print('Generation: {}.Loss = {}. Acc = {}.'.format(i+1,temp_loss,acc)) 

survived = []
prediction_d = np.array(test_data.ix[:,['Pclass','Parch','SibSp','Sex','Age','Embarked']].values).astype(np.float64)    
for i in range(len(prediction_d)):
    pre_x = np.array([prediction_d[i]])
    predict_survived = np.argmax(np.array(sess.run(f_output,feed_dict = {data:pre_x})))
    survived.append(predict_survived)

output = [['PassengerId','Survived']]
pid = 892
for i in range(len(survived)):
    output.append([pid,survived[i]])
    pid += 1
csvP.writeDataIntoCsv(output)

plt.plot(steps, loss, 'k--', label='Train Set')
plt.title('Loss')
plt.xlabel('Steps')
plt.ylabel('Softmax Loss')
plt.legend(loc='upper left')
plt.show()

