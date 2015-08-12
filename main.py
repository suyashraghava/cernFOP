import csv
import math
import numpy as np
import six
import toolDA as da
import  sys

# load the trainning file
train = csv.reader( open( r'training.csv') )

# convert the csv object to a list
train = [[i for i in y] for y in train]

#remove the unwanted collums
i = 0
while (i < ( len(train[0] ) )):

    if (train[0][i] == 'mass'):
        da.delCollumn(train,i)
        i = i - 4

    if ( train[0][i]== 'production'):
        da.delCollumn(train,i)
        i = i - 4

    if ( train[0][i] == 'min_ANNmuon'):
        da.delCollumn(train,i)
        i = i - 4
    if (train[0][i] == 'SPDhits'):
        da.delCollumn(train,i)

#get the signal collumn to predict
    if ( train[0][i] == 'signal'):
        signal = da.getCollumn(train,i)
        da.delCollumn(train,i)
        i = i - 4
    i = i + 1

#remove the header
train = [[ float(i) for i in y] for y in train[1:] ]
signal =  [[ float(i) for i in y] for y in signal[1:] ]

#now train
import sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(random_state = 0, n_estimators=60,max_depth=38)
clf.fit(train[1:],signal[1:])

#now predict
test = csv.reader( open(r'test.csv'))
c = csv.writer(open("submit1.csv","wb"))
c.writerow(['id','units'])
test = [[float(i) for i in y] for y in test[1:]]
for i in test[1:]:
    c.writerow( [i[0] , str( clf.predict(i) )] )






