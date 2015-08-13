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

#now train
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(random_state = 1, n_estimators=100)
clf.fit(train[1:],signal[1:])

#now predict
test = csv.reader( open(r'test.csv'))
c = csv.writer(open("submit1.csv","wb"))
c.writerow(['id','prediction'])
test = [[i for i in y] for y in test]
for i in range(len(test[0])):

    if ( test[0][i] == 'SPDhits'):
        da.delCollumn(test,i)
print len(test[1])
print len(train[1])
for i in test[1:]:

    c.writerow( [i[0] , str( clf.predict_proba(i) ) ] )






