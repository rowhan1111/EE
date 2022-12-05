import numpy as np
import random
import csv
import math
import time
# RelU function

datalist = []
#create the datapoints to be used
with open('cab_finweather.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        datalist.append(row)


def RelU(value):
    if value >= 0:
        return value
    else:
        return 0


# defining the derivative ReLU function
def derivRelU(value):
    if value >= 0:
        return 1
    else:
        return 0

#to output using the weights and inputs given
def output(inhid, hidout, inputs, targets, b1, bout, predict):
    #use the function if not predicting whether it is correct or not
    if predict == False:
        hidden = np.matmul(inputs, inhid) + b1

        derivhid = np.matrix(np.array(hidden))
        temp = []
        temp2 = []
        for f in range(0, hidden.size):
            temp.append(RelU(hidden.item(f)))
            temp2.append(derivRelU(derivhid.item(f)))
        hidden = np.matrix(temp)
        derivhid = np.matrix(temp2)

        out = np.matmul(hidden, hidout) + bout
        derivout = np.matrix(np.array(out))
        temp3 = []
        temp4 = []
        for f in range(0, out.size):
            temp3.append(derivRelU(out.item(f)))
            temp4.append(RelU(out.item(f)))

        derivout = np.matrix(temp3)
        out = np.matrix(temp4)
        error = targets - out
        return (hidden, out, error, derivout, derivhid)
    #used to predict and get the accuracy
    else:
        hidden = np.matmul(inputs, inhid) + b1
        temp = []
        for f in range(0, hidden.size):
            temp.append(RelU(hidden.item(f)))
        hidden = np.matrix(temp)
        out = np.matmul(hidden, hidout) + bout
        temp3 = []
        for f in range(0, out.size):
            temp3.append(RelU(out.item(f)))
        out = np.matrix(temp3)
        if (out.item(0) > out.item(1)) == (targets[0] > targets[1]):
            return True
        else:
            return False


columnhid1 = 2  # refers to how many hidden nodes in first hidden layer
rowhid1 = 2  # refers to how many input nodes
inhid1 = np.matrix([[random.uniform(0.5, 1) for f in range(0, columnhid1)] for c in
                    range(0, rowhid1)])  # weights connecting input to hidden

columnout = 2  # referes to how many output nodes
rowout = columnhid1  # refers to how many hidden nodes in hidden layer
hid1out = np.matrix([[random.uniform(0.5, 1) for f in range(0, columnout)] for c in
                     range(0, rowout)])  # weights connecting hidden to output
#bias and learning rates
b1 = 0.4
bout = 0.4
lr = 0.04

#function for back propagation
def backprop(inhid, hidout, inputs, hidden, derivhid, derivout, error):
    hidout_temp = np.matrix(np.array(hidout))
    li_out_to_lasthid = []
    columnhid = inhid.shape[1]
    rowhid = inhid.shape[0]
    # create a matrix containing the sum of -error * derivout * weights for 1 hidden node
    for i in range(columnhid):
        li_out_to_lasthid.append(np.sum(np.multiply(np.multiply(-error, derivout), hidout_temp[i, :])))
    ma_out_to_lasthid = np.matrix(li_out_to_lasthid)
    #do backprop for inhid1
    inhidtemp = []
    for i in range(rowhid):
        for f in range(columnhid):
            inhidtemp.append(inhid.item(i, f) - lr * derivhid.item(f) * ma_out_to_lasthid.item(f) * inputs.item(i))
    inhid1 = np.reshape(np.matrix(inhidtemp), (rowhid, columnhid), order="C")
    #working for hid1out
    hidouttemp = []
    rowout = hidout.shape[0]
    columnout = hidout.shape[1]
    for i in range(rowout):
        for f in range(columnout):
            hidouttemp.append(hidout.item(i, f) - lr * derivout.item(f) * hidden.item(i) * (-error.item(f)))
    hidout = np.reshape(np.matrix(hidouttemp), (rowout, columnout), order="C")
    return (inhid1, hidout)


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
        hidden, out, error, derivout, derivhid = output(inhid1, hid1out, inputs,
                                                        [float(i[0]), 1 - float(i[0])], b1, bout, False)
        inhid1, hid1out = backprop(inhid1, hid1out, inputs, hidden, derivhid, derivout, error)

        for f in testdataset:
            if output(inhid1, hid1out, [float(f[1]), float(f[2])], [float(f[0]), 1 - float(f[0])], b1, bout, True):
                accuracy += 1
        print("time:", time.time() - start)
        print("accuracy:", accuracy/lentest)
