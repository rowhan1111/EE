import numpy as np
import random
#RelU function
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
def output(inhid, hidout, inputs, targets):
    hidden = np.matmul(inputs, inhid)
    for f in range(0, len(hidden)-1):
        hidden[f] = RelU(hidden[f])
    #output layer function
    out = np.matmul(hidden, hidout)
    for f in range(0, len(out)-1):
        out[f] = RelU(out[f])
    error = targets-out
    return(hidden, out, error)

columnhid = 2#refers to how many hidden nodes
rowhid = 2#refersto how many input nodes
inhid = np.matrix([[random.uniform(0.5, 1) for f in range(0, columnhid)] for c in range(0, rowhid)])
columnout = 2 #referes to how many output nodes
rowout = columnhid#refers to how many hidden nodes
hidout = np.matrix([[random.uniform(0.5, 1) for f in range(0, columnout)] for c in range(0, rowout)])
#input & target matrix
inputs = np.matrix([1, 0])
targets = np.matrix([0, 1])


hidden, out, error = output(inhid, hidout, inputs, targets)
#print(hidden, out, error)

#print(inhid)
def backprop(inhid, hidout, error):
    global rowhid
    global rowout
    global columnhid
    global columnout
    
    temp = []
    lr = 0.5
    for c in range(rowhid * columnout):
        x = inhid.item(c)
        for f in range(1):
            x -= lr*(-error.item(f) * derivRelU(out.item(f)) * hidout.item(c%rowhid, f)) * derivRelU(hidden.item(c%rowhid)) * inputs.item(int(c >= rowhid))
        temp.append(x)
        #print(temp)
        
        if len(temp) == rowout:
            inhid[int((c-1)/rowout)] = temp
            temp = []

    #print(inhid)
    temp = []
    #print(hidout)

    for c in range(rowout * columnout):
        x2 = hidout.item(c) - lr*(-error.item(c%columnout) * derivRelU(out.item(c%columnout)) * hidden.item(int(c >=columnout)))
        temp.append(x2)
        #print(temp)
        
        if len(temp) == columnout:
            hidout[int((c-1)/columnout)] = temp
            temp = []

    return(inhid, hidout)
print(output(inhid, hidout, inputs, targets))
newweights = backprop(inhid, hidout, error)
print(output(newweights[0], newweights[1], inputs, targets))

