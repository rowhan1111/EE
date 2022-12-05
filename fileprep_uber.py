#code to prepare the data to be used to train the neural network
import csv
from datetime import datetime
import pandas as pd

datetime = []
distance = []
price = []
amount = 0
with open('cab_rides.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        #have the data be only for cars that are UberXL
        if row[2] != "time_stamp" and row[5] != "" and row[9] == "UberXL" and amount < 10000:
            df = pd.to_datetime(int(row[2]), unit = 'ms')
            x = (df.hour * 60 + df.minute)
            #[price, distance, time]
            datetime.append(x/1440)
            if x > 1440:
                print(x)
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
for i in datetime:

    #print(index, i)
    if ((price[index] - minprice)/diffprice) > temp:
        p = 0.99
        num += 1
    else:
        p = 0.01
    new.append([p, (distance[index] - mindist)/diffdistance, i])
    index += 1
#writing the data file
with open('cab_fin10000.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(new)
