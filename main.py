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


#just for a sake of testing it
signal = [int(i) for  i in signal[1:]]
train  = [[float(i) for i in y ] for  y in train[1:]]
#now train
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xg
gB =  GradientBoostingClassifier()
clf = RandomForestClassifier(random_state = 1, n_estimators=10)
var = {"objective": "binary:logistic",
                  "eta": 0.3,
                            "max_depth": 7,
                                      "min_child_weight": 3,
                                                "silent": 1,
                                                          "subsample": 0.7,
                                                                    "colsample_bytree": 0.7,
                                                                              "seed": 1}


gB.fit(train,signal)
clf.fit(train,signal)
xgTrain =  xg.train(var,xg.DMatrix(train,signal),300)





#now predict
test = csv.reader( open(r'test.csv'))
w = csv.writer(open("submit1.csv","wb"))
w.writerow(['id','prediction'])
test = [[i for i in y] for y in test]
for i in range(len(test[0])):

    if ( test[0][i] == 'SPDhits'):
        da.delCollumn(test,i)
test = [[ float(i) for  i in y]for  y in test[1:]]

for i in test:
    c = clf.predict_proba(i)
    b = gB.predict_proba(i)
    d = xgTrain.predict(xg.DMatrix([i]))

    w.writerow( [int(i[0]) , ((b[:,1][0]) + c[:,1][0] + d[0])/3]  )


##pr, tpr, thresholds = metrics.roc_curve(signal,p )
##score = 0
##for i in tpr :
##    if ( i <= 0.2 ) :
##        score += i * 0.2
##    if ( i>0.2 and i <= 0.4):
##        score += i * 1.5
##    if ( i > 0.4 and i <= 0.6):
##        score += i * 1.0
##    if ( i> 0.6 and i <= 0.8):
##        score += i * 0.5
##    if ( i > 0.8):
##        score += i * 0.0
#
#def auc_truncated(labels, predictions, tpr_thresholds=(0.2, 0.4, 0.6, 0.8),roc_weights=(4, 3, 2, 1, 0)):
#    assert numpy.all(predictions >= 0.) and numpy.all(predictions <= 1.), 'Data predictions are out of range [0, 1]'
#    assert len(tpr_thresholds) + 1 == len(roc_weights), 'Incompatible lengths of thresholds and weights'
#    fpr, tpr, _ = roc_curve(labels, predictions)
#    area = 0.
#    tpr_thresholds = [0.] + list(tpr_thresholds) + [1.]
#    for index in range(1, len(tpr_thresholds)):
#        tpr_cut = numpy.minimum(tpr, tpr_thresholds[index])
#        tpr_previous = numpy.minimum(tpr, tpr_thresholds[index - 1])
#        area += roc_weights[index - 1] * (auc(fpr, tpr_cut, reorder=True) - auc(fpr, tpr_previous, reorder=True))
#    tpr_thresholds = numpy.array(tpr_thresholds)
#    # roc auc normalization to be 1 for an ideal classifier
#    area /= numpy.sum((tpr_thresholds[1:] - tpr_thresholds[:-1]) * numpy.array(roc_weights))
#    return area
#
#
#print auc_truncated(signal,p,tpr_thresholds=(0.2, 0.4, 0.6, 0.8),roc_weights=(4, 3, 2, 1, 0))
#








