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
signal = [int(i) for  i in signal[1:]]
train  = [[float(i) for i in y ] y in train[1:]]
for i in signal[1:]:
    if ( i != 1 and  i != 0 ):
        print i
#now train
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state = 1, n_estimators=10)
clf.fit(train[1:],signal)

#now predict
test = csv.reader( open(r'test.csv'))
c = csv.writer(open("submit1.csv","wb"))
c.writerow(['id','prediction'])
test = [[i for i in y] for y in test]
for i in range(len(test[0])):

    if ( test[0][i] == 'SPDhits'):
        da.delCollumn(test,i)
for i in test[1:]:
    b = clf.predict_proba(i)
    c.writerow( [i[0] , (b[:,1][0])]  )

# Validate
from sklearn import metrics

p = []
for i in train[1:]:
    b = clf.predict_proba(i)
    p.append(b[:,1][0])


fpr, tpr, thresholds = metrics.roc_curve(signal,p )
score = 0
for i in tpr :
    if ( i <= 0.2 ) :
        score += i * 0.2
    if ( i>0.2 and i <= 0.4):
        score += i * 1.5
    if ( i > 0.4 and i <= 0.6):
        score += i * 1.0
    if ( i> 0.6 and i <= 0.8):
        score += i * 0.5
    if ( i > 0.8):
        score += i * 0.0

print score
print "complete"







