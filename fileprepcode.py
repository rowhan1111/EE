import numpy as np
import pandas
import csv
#nueral network comparing ReLU and Sigmoid

'''
#code used to process the data as it was kind of corrupted
data named: LOL_data.csv

templist = []
tierlist = ['God', 'S', 'A', 'B', 'C', 'D']

with open('LOL1212.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        temp = row[0].split(";")
        templist2 = []
        if temp[3] in tierlist:
            temp2 = tierlist.index(temp[3])
        else:
            temp2 = temp[3]
        
        templist2.extend([temp[0], temp2, temp[6], temp[8], temp[9]])
        templist.append(templist2)



with open('LOL_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(templist)
'''
#name, tier, win %, pick %, ban % || but going to use mainly tier, win %, ban %
tempdict = {}
tempdict2 = {}
index = 0
with open('LOL_data.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[0] not in list(tempdict.keys()):
            tempdict.update({row[0]: [row[1:]]})
            tempdict2.update({index: [row[4]]})
            index += 1
        else:
            x = tempdict[row[0]]
            x.append(row[1:])
            tempdict.update({row[0]: x})
tempdict2.pop(0)

for i in tempdict2:
    tempdict2.update({i: [float(tempdict2[i][0][:-1])]})

def Weightedavg(actvallist, weightlist):
    val = 0
    for i in actvallist:
        val += i
    weight = 0
    for f in weightlist:
        weight += f
        
    return([val/weight])
index = 0
for i in tempdict:
    if i != "Name":
        index += 1
        tierpicklist = []
        winpicklist = []
        picklist = []
        
        for f in tempdict[i]:
            tierpicklist.append(float(f[0]) * (float(f[2][:-1])))
            winpicklist.append(float(f[1][:-1]) * (float(f[2][:-1])))
            picklist.append(float(f[2][:-1]))

        tempdict2.update({index: Weightedavg(tierpicklist, picklist) + Weightedavg(winpicklist, picklist) + tempdict2[index]})
        
        
print(tempdict2)
winlist = []
banlist = []

for i in tempdict2:
    winlist.append(tempdict2[i][1])
    banlist.append(tempdict2[i][2])

windiff = max(winlist) - min(winlist)
minwin = min(winlist)
bandiff = max(banlist) - min(banlist)
minban = min(banlist)

findict = {}

index = 0
for i in tempdict2:
    if tempdict2[i][0] <= 2.5:
        tier = 0.99
    else:
        tier = 0.01
    findict.update({index: [tier, ((tempdict2[i][1] - minwin)/windiff), ((tempdict2[i][2] - minban)/bandiff)]})
    index += 1
    
#tier is 1 if equal or higher than B, 0 if lower than B(2< x)
    
finlist = [['Tier', 'Win', 'Ban']]
for i in findict:
    finlist.append(findict[i])
print(finlist)

with open('LOL_data_fin.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(finlist)
