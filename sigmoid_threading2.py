#neural network using ReLU function
#importing libraries that are needed
import csv
import random 
import math
import matplotlib.pyplot as plt
import threading

datalist = []
#getting the data from a .csv file
with open('cab_fin.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        datalist.append(row)
#setting the bias, learning rate   
b1 = 0.2
b2 = 0.2
lr = 0.02


#sigmoid function and its derivative
def sigmoid(value):
    return 1 / (1 + math.exp(-value))
def deriv(value):
    return value * (1 - value)
#a function to find the output of the neural network using the weights and the input data
def output(weight_list, index, datalist):
    w1 = weight_list[0]
    w2 = weight_list[1]
    w3 = weight_list[2]
    w4 = weight_list[3]
    w5 = weight_list[4]
    w6 = weight_list[5]
    w7 = weight_list[6]
    w8 = weight_list[7]
    i1 = float(datalist[index][1])
    i2 = float(datalist[index][2])
    
    h1 = i1 * w1 + i2 * w3 + b1
    h2 = i1 * w2 + i2 * w4 + b1
    o1 = sigmoid(h1) * w5 + sigmoid(h2) * w7 + b2
    o2 = sigmoid(h1) * w6 + sigmoid(h2) * w8 + b2
    
    if sigmoid(o1) > sigmoid(o2):
        return 0.99
    else:
        return 0.01
    
#function for back propagation
def back_prop(weight_list, index, datalist):
    w1 = weight_list[0]
    w2 = weight_list[1]
    w3 = weight_list[2]
    w4 = weight_list[3]
    w5 = weight_list[4]
    w6 = weight_list[5]
    w7 = weight_list[6]
    w8 = weight_list[7]
    
    
    i1 = float(datalist[index][1])
    i2 = float(datalist[index][2])
    actoutput1 = float(datalist[index][0])
    actoutput2 = 1 - actoutput1


    h1 = i1 * w1 + i2 * w3
    h2 = i1 * w2 + i2 * w4
    outh1 = sigmoid(h1)
    outh2 = sigmoid(h2)

    o1 = outh1 * w5 + outh2 * w7
    o2 = outh1 * w6 + outh2 * w8


    outo1 = sigmoid(o1)
    outo2 = sigmoid(o2)
    

    w5n = w5 - lr * (-(actoutput1 - outo1)) * deriv(outo1) * outh1
    w6n = w6 - lr * (-(actoutput2 - outo2)) * deriv(outo2) * outh1
    w7n = w7 - lr * (-(actoutput1 - outo1)) * deriv(outo1) * outh2
    w8n = w8 - lr * (-(actoutput2 - outo2)) * deriv(outo2) * outh2


    w1n = w1 - lr* (((-(actoutput1 - outo1) *deriv(outo1) * w5) + (-(actoutput2 - outo2) * deriv(outo2) * w6)) * deriv(outh1) * i1)
    w2n = w2 - lr* (((-(actoutput1 - outo1) *deriv(outo1) * w7) + (-(actoutput2 - outo2) * deriv(outo2) * w8)) * deriv(outh2) * i1)
    w3n = w3 - lr* (((-(actoutput1 - outo1) *deriv(outo1) * w5) + (-(actoutput2 - outo2) * deriv(outo2) * w6)) * deriv(outh1) * i2)
    w4n = w4 - lr* (((-(actoutput1 - outo1) *deriv(outo1) * w7) + (-(actoutput2 - outo2) * deriv(outo2) * w8)) * deriv(outh2) * i2)
    
    return([w1n, w2n, w3n, w4n, w5n, w6n, w7n, w8n])



totnumtest = 5000
percentage = 0.75
accmean = 0
#making the train set and the evaluation set
def maketest(origlist):
    templist = origlist[:]
    trainlist = []
    while (len(templist)/(totnumtest)) > 0.25:
        index = random.randrange(len(templist))
        trainlist.append(templist[index])
        templist.pop(index)
        
    return ([trainlist, templist])

#training the neural network
for i in range(0, 10):
    a = 0.5
    b = 1

    w1 = random.uniform(a, b)
    w2 = random.uniform(a, b)
    w3 = random.uniform(a, b)
    w4 = random.uniform(a, b)
    w5 = random.uniform(a, b)
    w6 = random.uniform(a, b)
    w7 = random.uniform(a, b)
    w8 = random.uniform(a, b)


    weight_list = [w1, w2, w3, w4, w5, w6, w7, w8]
    num_list = []
    accuracy_list = []

    correct = 0
    num = 0
    traindataset = maketest(datalist)

    time = 0
    for i in range(int(totnumtest * percentage)):
        correct = []
        templist = []
        threads = []
        def check(index):
            for f in range(int(index), int(index + totnumtest*(1-percentage)/10)):
                out = output(weight_list, f, traindataset[1])
                if out == float(traindataset[1][f][0]):
                    correct.append(1)
        for f in range(10):
            th = threading.Thread(target=check, args=((f*int(totnumtest * (1-percentage))/10),))
            threads.append(th)
        for thread in threads:
            thread.start()
        num += 1
        num_list.append(num)
        accuracy_list.append(len(correct)/((1-percentage)*totnumtest))
        weight_list = back_prop(weight_list, i, traindataset[0])

    print(accuracy_list[0], accuracy_list[-1], max(accuracy_list))
    print(weight_list)
    accmean += max(accuracy_list)
print(accmean/10)


plt.plot(num_list, accuracy_list)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.ylim([0, 1])

plt.show()


            


