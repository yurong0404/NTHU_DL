# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# library
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#1-1&1-2~prepocessing

data = pd.read_csv('train.csv')
data = pd.get_dummies(data)
data_no = pd.read_csv('test_no_G3.csv')
data_no = pd.get_dummies(data_no)
training_data = data.iloc[0 :800]
testing_data =data.iloc[800:1000]
print ('training_data:',training_data)
print ('testing_data:',testing_data)
for item in data.columns  :
    if item != 'ID' and item != 'G3':
        testing_data[item] = (testing_data[item]-training_data[item].mean())/(training_data[item].std())
for item in data_no.columns  :
    if item != 'ID' :
        data_no[item] = (data_no[item]-training_data[item].mean())/(training_data[item].std())
for item in data.columns:
    if item != 'ID' and item != 'G3':   
        training_data[item] = (training_data[item]-training_data[item].mean())/(training_data[item].std())
G3_list=training_data['G3'].values
G3_list_test=testing_data['G3'].values
A=training_data.drop(columns=['ID', 'G3']).values
A_test=testing_data.drop(columns=['ID', 'G3']).values

  #Drop rows df.drop([0, 1])

#1-2   
x_ls = np.dot(np.linalg.pinv(A),np.transpose(G3_list))
G3_test_ls = np.dot(A_test,np.transpose(x_ls))
RMES = np.sqrt(((G3_test_ls - G3_list_test) ** 2).mean())
print ('RMES:',RMES)

#1-3
x_ls_reg = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(A),A)+1*np.identity(60)),np.transpose(A)),np.transpose(G3_list))
# W=(AtA+lamda*L)^-1+At*y
G3_test_ls_reg = np.dot(A_test,np.transpose(x_ls_reg))
RMES_reg =  np.sqrt(((G3_test_ls_reg - G3_list_test) ** 2).mean())
print ('RMES_reg:',RMES_reg)


#1-4
temp = np.ones(800)
A_bias = np.c_[A,temp]
temp = np.ones(200)
A_test_bias = np.c_[A_test,temp]
x_ls_rb = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(A_bias),A_bias)+1*np.identity(61)),np.transpose(A_bias)),np.transpose(G3_list))
G3_test_ls_rb = np.dot(A_test_bias,np.transpose(x_ls_rb))
RMES_rb =  np.sqrt(((G3_test_ls_rb - G3_list_test) ** 2).mean())
print ('RMES_rb:',RMES_rb)


#1-5
x_ls_Bayesian = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(A_bias),A_bias)+1*np.identity(61)),np.transpose(A_bias)),np.transpose(G3_list))
G3_test_ls_Bayesian = np.dot(A_test_bias,np.transpose(x_ls_rb))
RMES_Bayesian =  np.sqrt(((G3_test_ls_rb - G3_list_test) ** 2).mean())
print('RMES_Bayesian:',RMES_Bayesian)
#1-6
plt.plot(testing_data["ID"],G3_list_test,'b',label='ground truth')
plt.plot(testing_data["ID"],G3_test_ls,'k',label='linear regrssion')
plt.plot(testing_data["ID"],G3_test_ls_reg,'r',label='linear regression reg')
plt.plot(testing_data["ID"],G3_test_ls_rb,'y',label='linear regrssion r/b')
plt.plot(testing_data["ID"],G3_test_ls_rb,'c',label='linear regrssion Bayesian')
plt.legend(loc='lower right')
#plt.tight_layout()
#2-1
G3_cls = np.ones(800)
for i in range(0,799):
    if G3_list[i]>=10:
        G3_cls[i] = 1
    else :
        G3_cls[i] = 0 
G3_cls_test = np.ones(200)
for i in range(0,199):
    if G3_list_test[i]>=10:
        G3_cls_test [i] = 1
    else :
         G3_cls_test[i] = 0 
temp = np.ones(800)
A_bias_cls = np.c_[A,temp]
temp = np.ones(200)
A_test_bias_cls = np.c_[A_test,temp]

x_rb_cls = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(A_bias_cls),A_bias_cls)+1*np.identity(61)),np.transpose(A_bias_cls)),np.transpose(G3_cls))
G3_test_cls = np.dot(A_test_bias_cls,np.transpose(x_rb_cls))

th1=0
th5=0
th9=0
for i in range(0,199):
    if G3_cls_test[i]==1:
        if G3_test_cls[i]>=0.1:
            th1 += 1
        if G3_test_cls[i]>=0.5:
            th5 += 1
        if G3_test_cls[i]>=0.9:
            th9 += 1
    else :
        if G3_test_cls[i]<0.1:
            th1 += 1
        if G3_test_cls[i]<0.5:
            th5 += 1
        if G3_test_cls[i]<0.9:
            th9 += 1 
print ('th=0.1',th1/200.0,'th=0.5',th5/200.0,'th=0.9',th9/200.0  )

#2-2         
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(features, weights):
    
  z = np.dot(features, weights)
  return sigmoid(z)

def update_weights(features, labels, weights, lr):

    N = len(features)
    predictions = predict(features, weights)
    gradient = np.dot(features.T,  predictions - labels)
    gradient /= N
    gradient *= lr
    weights -= gradient

    return weights

def cost_function(features, labels, weights):
  
    observations = len(labels)
    predictions = predict(features, weights)
    class1_cost = -labels*np.log(predictions)
    class2_cost = -(1-labels)*np.log(1-predictions)
    cost = class1_cost + class2_cost
    cost = cost.sum() / observations
    return cost

def train(features, labels, weights, lr, iters):

    for i in range(iters):
        weights = update_weights(features, labels, weights, lr)

        #Calculate error for auditing purposes
        cost = cost_function(features, labels, weights)
        
        # Log Progress
        if i % 100 == 0:
            print( "iter: "+str(i) + " cost: "+str(cost))

    return weights

weight = np.ones(61)
temp = np.ones(800)
A_lg = np.c_[A,temp]
temp = np.ones(200)
A_test = np.c_[A_test,temp]
train(A_lg , G3_cls ,weight, 0.1 , 3000)
copy = predict(A_test , weight)

th1_=0
th5_=0
th9_=0
for i in range(0,199):
    if G3_cls_test[i]==1:
        if copy[i]>=0.1:
            th1_ += 1
        if copy[i]>=0.5:
            th5_ += 1
        if copy[i]>=0.9:
            th9_+=1
    else:
        if copy[i]<0.1:
            th1_ += 1
        if copy[i]<0.5:
            th5_ += 1
        if copy[i]<0.9:
            th9_+=1
print ('th=0.1',th1_/200.0,'th=0.5',th5_/200.0,'th=0.9',th9_/200.0  )


#2-3
temp1 = np.zeros([2,2])
temp2 = np.zeros([2,2])
for i in range(0,199):
    if  G3_cls_test[i]==1:
        if copy[i]>=0.5:
            temp2[1,1]+=1
        if copy[i]<0.5:
            temp2[0,1]+=1
        if G3_test_cls[i]>=0.5:
            temp1[1,1]+=1
        if G3_test_cls[i]<0.5:
            temp1[0,1]+=1
    if  G3_cls_test[i]==0:
        if copy[i]>=0.5:
            temp2[1,0]+=1
        if copy[i]<0.5:
            temp2[0,0]+=1
        if G3_test_cls[i]>=0.5:
            temp1[1,0]+=1
        if G3_test_cls[i]<0.5:
            temp1[0,0]+=1   
            

#plt.figure(figsize = (10,7))
f,(ax1) = plt.subplots(1,1,sharey=False)
df_cm = pd.DataFrame(temp2 , ["predict=0","predict=1"],["true=0","true=1"])
sn.heatmap(df_cm, annot=True)
ax1.set_title('Thrshold=0.5 / logistic regression')
f,(ax1) = plt.subplots(1,1,sharey=False)
df_cm2 = pd.DataFrame(temp1 ,["predict=0","predict=1"],["true=0","true=1"])
sn.heatmap(df_cm2, annot=True)
ax1.set_title('Thrshold=0.5 / linear regression')
#2-4
temp3 = np.zeros([2,2])
temp4 = np.zeros([2,2])
for i in range(0,199):
    if  G3_cls_test[i]==1:
        if copy[i]>=0.9:
            temp4[1,1]+=1
        if copy[i]<0.9:
            temp4[0,1]+=1
        if G3_test_cls[i]>=0.9:
            temp3[1,1]+=1
        if G3_test_cls[i]<0.9:
            temp3[0,1]+=1
    if  G3_cls_test[i]==0:
        if copy[i]>=0.9:
            temp4[1,0]+=1
        if copy[i]<0.9:
            temp4[0,0]+=1
        if G3_test_cls[i]>=0.9:
            temp3[1,0]+=1
        if G3_test_cls[i]<0.9:
            temp3[0,0]+=1   
            

#plt.figure(figsize = (10,7))
f,(ax3) = plt.subplots(1,1,sharey=False)

df_cm3 = pd.DataFrame(temp4 , ["predict=0","predict=1"],["true=0","true=1"])
sn.heatmap(df_cm3, annot=True)
ax3.set_title('Thrshold=0.9 / logistic regression')

f,(ax3) = plt.subplots(1,1,sharey=False)
df_cm4 = pd.DataFrame(temp3 , ["predict=0","predict=1"],["true=0","true=1"])
sn.heatmap(df_cm4, annot=True)
ax3.set_title('Thrshold=0.9 / linear regression')
#2-5

#3-1


A_no = data_no.drop(columns=['ID']).values
temp = np.ones(44)
A_no = np.c_[A_no,temp]
x_ls_rb_no=np.delete(x_ls_rb, 41)
x_ls_rb_no=np.delete(x_ls_rb_no, 32)

G3_test_ls_rb_no = np.dot(A_no , np.transpose(x_ls_rb_no))

print ('3-1',G3_test_ls_rb_no)

#3-2

weight_no = np.ones(59)
temp = np.ones(800)
A_lg_no = np.c_[A,temp]
A_lg_no = np.delete(A_lg_no, 41, 1)
A_lg_no = np.delete(A_lg_no, 32, 1)

weight_no = train(A_lg_no , G3_cls , weight_no , 0.1 , 3000)
copy_no = predict(A_no,weight_no)
 
print ('3-2',copy_no)

#3-3
fp1 = open("107062604_1.txt", "w")
 
for i in range(1001,1045):
    fp1.write(str(i)+'\t'+str(G3_test_ls_rb_no[i-1001])+'\n')
 
fp2 = open("107062604_2.txt", "w")

for i in range(1001,1045):
    fp2.write(str(i)+'\t'+str(copy_no[i-1001])+'\n')