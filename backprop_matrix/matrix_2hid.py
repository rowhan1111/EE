import numpy as np
import random
import csv
import time
import math

datalist = []
#create the datapoints to be used
with open('cab_finweather.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        datalist.append(row)

#RelU function
def RelU(value):
    if value >0:
        return value
    else:
        return 0
#defining the derivative ReLU function
def derivRelU(value):
    if value > 0:
        return 1
    else:
        return 0
#the closer to output, the lower the value will be
def output(inhid2, hid2hid1, hid1out, b2, b1, bout, inputs, targets, predict):
    if predict==False:
        hidden2 = np.matmul(inputs, inhid2) + b2
        derivhid2 = np.matrix(np.array(hidden2))
        temp = []
        temp2 = []
        for f in range(0, hidden2.size):
            temp.append(RelU(hidden2.item(f)))
            temp2.append(derivRelU(derivhid2.item(f)))
        hidden2 = np.matrix(temp)
        derivhid2 = np.matrix(temp2)
        #print("hid2hid1:", hid2hid1)
        #print("hidden2:", hidden2)
        hidden1 = np.matmul(hidden2, hid2hid1) + b1
        derivhid1 = np.matrix(np.array(hidden1))
        temp = []
        temp2 = []
        for f in range(0, hidden1.size):
            temp.append(RelU(hidden1.item(f)))
            temp2.append(derivRelU(derivhid1.item(f)))
        hidden1 = np.matrix(temp)
        derivhid1 = np.matrix(temp2)
        
        
        out = np.matmul(hidden1, hid1out) + bout
        derivout = np.matrix(np.array(out))
        temp = []
        temp2 = []
        for f in range(0, out.size):
            temp.append(derivRelU(out.item(f)))
            temp2.append(RelU(out.item(f)))
        
        derivout = np.matrix(temp)
        out = np.matrix(temp2)
        error = targets-out
        return(hidden2, hidden1, out, error, derivhid2, derivhid1, derivout)
    else:
        hidden2 = np.matmul(inputs, inhid2) + b2
        temp = []
        for f in range(0, hidden2.size):
            temp.append(RelU(hidden2.item(f)))
        hidden2 = np.matrix(temp)
        hidden1 = np.matmul(hidden2, hid2hid1) + b1
        temp = []
        for f in range(0, hidden1.size):
            temp.append(RelU(hidden1.item(f)))
        hidden1 = np.matrix(temp)        
        out = np.matmul(hidden1, hid1out) + bout
        temp = []
        for f in range(0, out.size):
            temp.append(RelU(out.item(f)))
        out = np.matrix(temp)
        if (out.item(0) > out.item(1)) == (targets[0] > targets[1]):
            return True
        else:
            return False

columnhid2 = 2#refers to how many hidden nodes in 2nd hidden layer
rowhid2 = 2#refers to how many input nodes
inhid2 = np.matrix([[random.uniform(0.5, 1) for f in range(0, columnhid2)] for c in range(0, rowhid2)])#weights connecting input to hidden
rowhid1 = columnhid2#how many hidden nodes in 2nd layer
columnhid1 = 2#refers to how many nodes in 1st hidden layer
hid2hid1 = np.matrix([[random.uniform(0.5, 1) for f in range(0, columnhid1)] for c in range(0, rowhid1)])

columnout = 2 #referes to how many output nodes
rowout = columnhid1#refers to how many hidden nodes in third hidden layer
hid1out = np.matrix([[random.uniform(0.5, 1) for f in range(0, columnout)] for c in range(0, rowout)])#weights connecting hidden to output
#bias and learning rates
b2 = 0.4
b1 = 0.4
bout = 0.4
lr = 0.04

def backprop(inhid2, hid2hid1, hid1out, inputs, hidden2, hidden1, derivhid2, derivhid1, derivout, error):

    hid1out_temp = np.matrix(np.array(hid1out))
    li_out_to_hid1 = []
    hid2hid1_temp = np.matrix(np.array(hid2hid1))
    li_hid1_to_hid2 = []
    
    columnhid2 = inhid2.shape[1]
    rowhid2 = inhid2.shape[0]
    columnhid1 = hid2hid1.shape[1]
    rowhid1 = hid2hid1.shape[0]
    rowout = hid1out.shape[0]
    columnout = hid1out.shape[1]
    
    #create a matrix containing the sum of -error * derivout * weights for 1st in the back hidden layer
    for i in range(columnhid1):
        li_out_to_hid1.append(np.sum(np.multiply(np.multiply(-error, derivout), hid1out_temp[i,:])))
    ma_out_to_hid1 = np.matrix(li_out_to_hid1)
    #create a matrix containing the sum of -error * derivout * weights for 1st in the back hidden layer
    
    for i in range(columnhid2):
        li_hid1_to_hid2.append(np.sum(np.multiply(np.multiply(ma_out_to_hid1, derivhid1), hid2hid1_temp[i,:])))
    ma_hid1_to_hid2 = np.matrix(li_hid1_to_hid2)
    #place for backprop 2nd layer
    inhid2temp = []
    for i in range(rowhid2):
        for f in range(columnhid2):
            inhid2temp.append(inhid2.item(i, f) - lr * derivhid1.item(f) * ma_hid1_to_hid2.item(f) * inputs.item(i))
    inhid2 = np.reshape(np.matrix(inhid2temp), (rowhid2, columnhid2), order="C")
    #1st hidden layer backprop(1st to the back)

    hid2hid1temp = []
    for i in range(rowhid1):
        for f in range(columnhid1):
            hid2hid1temp.append(hid2hid1.item(i, f) - lr * derivhid1.item(f) * ma_out_to_hid1.item(f) * hidden2.item(i))
    hid2hid1 = np.reshape(np.matrix(hid2hid1temp), (rowhid1, columnhid1), order="C")
    
    #output layer backprop
    hid1outtemp = []
    
    for i in range(rowout):
        for f in range(columnout):
            hid1outtemp.append(hid1out.item(i, f) - lr * derivout.item(f) * hidden1.item(i) * (-error.item(f)))
    hid1out = np.reshape(np.matrix(hid1outtemp), (rowout, columnout), order="C")
    return(inhid2, hid2hid1, hid1out)

totnumtest = len(datalist)
print(totnumtest)
percentage = 0.75


# creating the train set and the evaluation set
def maketest(origlist):
    templist = origlist[:]
    trainlist = []
    while (len(templist) / (totnumtest)) > (1 - percentage):
        index = random.randrange(len(templist))
        trainlist.append(templist[index])
        templist.pop(index)

    return ([trainlist, templist])

#creating the dataset, splitting them
x = maketest(datalist)
traindataset = x[0]
lengthtrain = len(traindataset)
testdataset = x[1]
lentrainshort = (math.floor(lengthtrain/100))
fintrain = []
lentest = len(testdataset)
x = 0

for i in range(100):
    fintrain.append(traindataset[lentrainshort * i:lentrainshort *(i+1)])
    x+= len(traindataset[lentrainshort * i:lentrainshort*(i+1)])
fintrain.append(traindataset[x:])

for c in fintrain:
    for i in c:
        accuracy = 0
        start = time.time()
        inputs = np.array([float(i[1]), float(i[2])])
        hidden2, hidden1, out, error, derivhid2, derivhid1, derivout = output(inhid2, hid2hid1, hid1out, b2, b1, bout, inputs,
                                                        [float(i[0]), 1 - float(i[0])], False)
        inhid2, hid2hid1, hid1out = backprop(inhid2, hid2hid1, hid1out, inputs, hidden2, hidden1, derivhid2, derivhid1, derivout, error)
        for f in testdataset:
            if output(inhid2, hid2hid1, hid1out, b2, b1, bout, [float(f[1]), float(f[2])], [float(f[0]), 1 - float(f[0])], True):
                accuracy += 1
        print("time:", time.time() - start)
        print("accuracy:", accuracy/lentest)


