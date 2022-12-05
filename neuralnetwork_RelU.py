#neural network using ReLU function
#importing libraries that are needed
import csv
import random 
import math
import matplotlib.pyplot as plt

#getting the data from a file
datalist = []
with open('cab_fin10000.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        datalist.append(row)


b1 = 0.2
b2 = 0.2
lr = 0.2
desired = 0.9

#defining the ReLU function
def RelU(value):
    if value >=0:
        return value
    else:
        return 0
#defining the derivative ReLU function
def derivRelU(value):
    if value >= 0:
        return 1
    else:
        return 0
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
    o1 = RelU(h1) * w5 + RelU(h2) * w7 + b2
    o2 = RelU(h1) * w6 + RelU(h2) * w8 + b2
    if RelU(o1) >= RelU(o2):
        return 0.99
    else:
        return 0.01

#function for back_propagation
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


    h1 = i1 * w1 + i2 * w3 + b1
    h2 = i1 * w2 + i2 * w4 + b1
    
    outh1 = RelU(h1)
    outh2 = RelU(h2)
    
    o1 = outh1 * w5 + outh2 * w7 + b2
    o2 = outh1 * w6 + outh2 * w8 + b2


    outo1 = RelU(o1)
    outo2 = RelU(o2)
    

    w5n = w5 - lr * (-(actoutput1 - outo1)) * derivRelU(outo1) * outh1
    w6n = w6 - lr * (-(actoutput2 - outo2)) * derivRelU(outo2) * outh1
    w7n = w7 - lr * (-(actoutput1 - outo1)) * derivRelU(outo1) * outh2
    w8n = w8 - lr * (-(actoutput2 - outo2)) * derivRelU(outo2) * outh2


    w1n = w1 - lr * (((-(actoutput1 - outo1) * derivRelU(outo1) * w5) + (-(actoutput2 - outo2) * derivRelU(outo2) * w6)) * derivRelU(outh1) * i1)
    w2n = w2 - lr * (((-(actoutput1 - outo1) * derivRelU(outo1) * w7) + (-(actoutput2 - outo2) * derivRelU(outo2) * w8)) * derivRelU(outh2) * i1)
    w3n = w3 - lr * (((-(actoutput1 - outo1) * derivRelU(outo1) * w5) + (-(actoutput2 - outo2) * derivRelU(outo2) * w6)) * derivRelU(outh1) * i2)
    w4n = w4 - lr * (((-(actoutput1 - outo1) * derivRelU(outo1) * w7) + (-(actoutput2 - outo2) * derivRelU(outo2) * w8)) * derivRelU(outh2) * i2)
    
    return([w1n, w2n, w3n, w4n, w5n, w6n, w7n, w8n])


totnumtest = len(datalist)
print(totnumtest)
percentage = 0.75
#creating the train set and the evaluation set
def maketest(origlist):
    templist = origlist[:]
    trainlist = []
    while (len(templist)/(totnumtest)) > (1-percentage):
        index = random.randrange(len(templist))
        trainlist.append(templist[index])
        templist.pop(index)
        
    return ([trainlist, templist])

a = 0.5
b = 1
accmean = 0
#training the neural network
acclist = []
initialweight = []

for z in range(0, 1):
    w1 = random.uniform(a, b)
    w2 = random.uniform(a, b)
    w3 = random.uniform(a, b)
    w4 = random.uniform(a, b)
    w5 = random.uniform(a, b)
    w6 = random.uniform(a, b)
    w7 = random.uniform(a, b)
    w8 = random.uniform(a, b)


    weight_list = [w1, w2, w3, w4, w5, w6, w7, w8]
    initialweight = weight_list[:]
    num_list = []
    accuracy_list = []

    correct = 0
    num = 0
    traindataset = maketest(datalist)

    time = 0
    for i in range(int(totnumtest * percentage)):
        correct = 0
        templist = []
        for f in range(int(totnumtest * (1-percentage))):
            out = output(weight_list, f, traindataset[1])
            if out == float(traindataset[1][f][0]):
                correct += 1
        num += 1
        num_list.append(num)
        accuracy_list.append(correct/((1-percentage)*totnumtest))
        weight_list = back_prop(weight_list, i, traindataset[0])
        print(i)
        print(weight_list)
        print(accuracy_list[-1])
        
    #print(accuracy_list[0], accuracy_list[-1], max(accuracy_list))
    #print(weight_list)
    accmean += max(accuracy_list)
    acclist.append(accmean)
print(accmean/100)
print(acclist)
print(initialweight)


plt.plot(num_list, accuracy_list)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.ylim([0, 1])

plt.show()


