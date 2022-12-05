import csv
from datetime import datetime
import pandas as pd


rainlist = []
datetimerain = []
temprain = 0
index = 0
numadded = 0
timeprev = "1545003901"
with open('weather.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        rain = 0
        if index > 0:
            if row[5] != timeprev:
                #print(pd.to_datetime(int(row[5]), unit = 's'))
                rainlist.append(temprain/numadded)
                datetime = (pd.to_datetime(int(row[5]), unit = 's'))
                time = [datetime.month, datetime.day, datetime.hour]
                datetimerain.append(time)
                timeprev = row[5]
                temprain = 0
                numadded = 0
            x = row[4]          
            if row[4] == "":
                x = 0
            temprain += float(x)
            numadded += 1
                
        index += 1


datetimeli = []
distance = []
price = []
amount = 0
with open('cab_rides.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        #have the data be only for cars that are UberXL
        if row[2] != "time_stamp" and row[5] != "" and row[9] == "UberXL":
            datetime = (pd.to_datetime(int(row[2]), unit = 'ms'))
            time = [datetime.month, datetime.day, datetime.hour]
            #print(time)
            datetimeli.append(time)
            distance.append(float(row[0]))
            price.append(float(row[5]))
            amount += 1
            
#taking the weighted average for the distance and the price

mindist = min(distance)
diffdistance = max(distance) - mindist

minprice = min(price)
diffprice = max(price) - minprice
temp = ((16 - minprice)/diffprice)
print(len(distance))
new = []
index = 0
num = 0


for i in datetimeli:

    #print(index, i)
    if ((price[index] - minprice)/diffprice) > temp:
        p = 0.99
        num += 1
    else:
        p = 0.01
    if i in datetimerain:
        rain = rainlist[datetimerain.index(i)]
        #print(rain)
    
        new.append([p, (distance[index] - mindist)/diffdistance, rain])
    index += 1

print(len(new))
#writing the data file
with open('cab_fin_max.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(new)

    
