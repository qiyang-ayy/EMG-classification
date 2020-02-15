# -*- coding: utf-8 -*-
"""
emgClassifier - Evaluation of Methods in Classification of Gestures from Electromyography Signal
This is a methodology to make the classification of gestures from EMG signals.
First, the pre-processing method for de-noising data via wavelet function
Second, the EMG signals are extracted to generate feature matrix, and machine learning methods are
used to classify the gestures from the data to verify the accuracy of those methods.

Class:
emgWave - read EMG file, create EMG classification dictionary, save EMG wave data from each sample,
            batch all samples
wt - wavelet transfer
featureExtract - Generate feature matrix, extracting features in time domain, frequency domain 
                    and time-frequency domain from emg data
MLs - Machine Learning methods, for verifying method validity

Data:
These are files of raw EMG data recorded by MYO Thalmic bracelet around the forearm. 
data files has 72 txt files with 8 nodes, which 40000-50000 recordings are in each node (column)
classifications of movement: hand at rest (1), hand clenched in a fist (2), wrist flexion
(3), wrist extension (4), radial deviations (5), and ulnar deviations (6)
    
Created on Thu Feb 13 19:25:54 2020
@author: Qiyang Ma
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pywt
import glob

'''
class emgWave - read EMG file, create EMG classification dictionary, save EMG wave data from each sample,
                batch all samples

Inputs:
filename - raw EMG signal data or de-noised EMG signal data -> str
timeSize - for filtering data which are less than timeSize -> int
    
Outputs:
t, mag. label - time series, magnitudes of nodes and classification label of EMG data -> list
dic - EMG classification dictionary -> dict
txt file - EMG wave data ( Original wave data and de-noised data ) -> .txt

Function:
read - read EMG files
labelSplit - dived original data into different classes and save them into a dictionary,
            filter data sets with small time scales ( less than timesize )
draw - present the waveforms of raw data or de-noised data of one or more nodes
save - save EMG data into files of different classes of each person
saveTxt - batch EMG data of all samples
'''

class emgWave:
    def __init__( self ):
        return
    
    def read( self, filename ):
#       t - time series of EMG data
#       mag - magnitude of raw EMG data or magnitude of de-noised EMG data
#       label - classification of gestures from EMG signal
        fr = open( filename )
        lines = fr.readlines()
        lines = lines[1:]
        t, mag, label = [], [], []
        for line in lines:
            tmp = line.split()
            t.append( tmp[0] )
            mag.append( tmp[1:-1] )
            label.append( tmp[-1] )
        return self.t, self.mag, self.label
    
    def labelSplit( self, t, mag, label, timeSize ):
#       timeSize - filter the small scale size data ( less than timeSize )
#       dic - save variables ( t, mag, label ) of different classes
        dic = {}
        splitPoint = [ 0 ]
        labelTab = []
        self.timeSize = timeSize
        n = 0
        while n < len( label ) - 1:
            if label[ n ] != label[ n+1 ]:
                splitPoint.append( n+1 )
                if label[ n ] not in labelTab:
                    labelCount = 1
                    labelTab.append( label[ n ] )
                    tmp = label[ n ] + '_' + str( labelCount )
                else:
                    labelTab.append( label[n] )
                    labelCount = labelTab.count( label[n] )
                    tmp = label[ n ] + '_' + str( labelCount )
                if ( splitPoint[-1] - splitPoint[-2] ) > self.timeSize:
                    index = slice( splitPoint[-2], (splitPoint[-1]), 1 )
                    dic[ tmp ] = [ t[ index ], mag[ index ], label[ index ] ]
            n += 1
        return self.dic
    
    def draw( self, t, nodes, *args ):
#       args - [ mag, fmag ], mag: magnitudes of raw EMG data; fmag: magnitudes of de-noised EMG data 
#       nodes - present the waveforms of one or more nodes simultaneously
        N = len( nodes )
        plt.figure( figsize = ( 10,8 ) )
        for i in range( N ):
            plt.subplot( N, 1, i+1 )
            for mag in args:
                tmp = [ float(x[0]) for x in mag ]
                plt.plot( t, tmp )
    
    def save( self, dic, Tag, option ):
#       option - original ( raw data ) or wavelet family ( de-noised data )
        if option == 'original':
            suffix = '_raw.txt'
        else:
            suffix = '_wt.txt'
        for key in dic.keys():
            values = dic[key]
            filename = key + Tag + suffix
            t, mag, label = np.mat( values[0]).T, np.mat( values[1] ), np.mat( values[2] ).T
            if option != 'original':
                w = wt()
                mag, _ = w.waveTransfer( mag, option )
            tmp = np.hstack( ( t, mag, label ) )
            np.savetxt( filename, tmp, fmt='%f %1.2e %1.2e %1.2e %1.2e %1.2e %1.2e  %1.2e %1.2e %1d' )
    
    def saveTxt( self, timeSize, option = 'orignal' ):
#       main function - generate txt file of raw data or de-noised data 
        for n in range( 1, 37 ):
            if n < 10:
                tmp = '0' + str( n )
            else:
                tmp = str( n )
            for i in range( 1, 3 ):
                if option == 'orignal':
                    fileGuide = tmp + '/' + str(i) + '*.txt'
                else:
                    fileGuide = tmp + '/' + tmp + '_' + str(i) + '*.txt'
                filename = glob.glob( fileGuide )
                t, mag, label = self.read( filename[0] )
                dic = self.labelSplit( t, mag, label, timeSize )
                Tag = '_' + tmp + '_' + str(i)
                self.save( dic, Tag, option )

'''
class wt - wavelet transfer

Inputs:
mag - magnitude of raw data or de-noised data -> list
family, threshold - family and threshold of wavelet -> str, float
**kwargs - decomposition level range -> dict = { floor = , upper = }
timeSize - for filtering data which are less than timeSize -> int

Outputs:
dataWt - de-noised data -> list
txt file - de-noised EMG wave data file in time domain and frequency domain -> .txt

Function:
waveTransfer - transfer raw data to de-noised data
saveFreq - save de-noised EMG data in frequency domain
freqTxt - batch saving de-noised data in frequency domain 
timeTxt - batch saving de-noised EMG data in time domain
'''

class wt:
    def __init__( self ):
        return
    
    def waveTransfer( self, mag, family = 'db6', threshold = 0.1, **kwargs ):
        self.dataWt = []
        self.f
        for i in range(8):
            emg1 = [ float( tmp[i] ) for tmp in mag ]
            w = pywt.Wavelet( family )
            maxlev = pywt.dwt_max_level( len(emg1), w.dec_len )            
            coeffs = pywt.wavedec( emg1, 'db6', level = maxlev )
            for i in range( 1, len( coeffs ) ):
                coeffs[i] = pywt.threshold( coeffs[i], threshold*max( coeffs[i] ) )    
            for j in range( 0, kwargs['floor'] ):
                for i in range( 0, len(coeffs[j]) ):
                    coeffs[j][i] = 0
            for j in range( kwargs['upper'], len(coeffs) ):
                for i in range( 0, len(coeffs[j]) ):
                    coeffs[j][i] = 0 
            emgRev = pywt.waverec( coeffs, family )
            plt.figure()
            plt.plot(emgRev)
            self.dataWt.append( emgRev[:len(emg1)] )
            self.f.append( abs( coeffs[4] ) )
        return self.dataWt, self.f
    
    def saveFreq( self, mag, label, Tag ):
#       save de-noised EMG data in frequency domain
        filename = Tag + '_f.txt'
        _, f = self.waveTransfer( mag, 'db6' )
        f = np.mat( f ).T
        n = np.mat( range( len( f ) ) ).T
        label = np.ones( (len(f), 1) ) * label[0]
        ftmp = np.hstack( ( n, f, label ) )
        np.savetxt( filename, ftmp, fmt='%d %1.2e %1.2e %1.2e %1.2e %1.2e %1.2e  %1.2e %1.2e %1d' )
    
    def freqTxt( self ):
#       batch saving de-noised data in frequency domain 
        for i in range( 1, 7 ):
            for j in range( 1, 3 ):
                for k in range(1,37):
                    if k < 10:
                        tmp = '0' + str( k )
                    else:
                        tmp = str( k )
                    for l in range( 1, 3 ):
                        Tag = str(i) + '_' + str(j) + '_' + tmp + '_' + str(l)
                        fileGuide = str(i) + '/' + Tag + '*.txt'
                        filename = glob.glob( fileGuide )
                        ew = emgWave()
                        _, mag, label = ew.read( filename[0] )
                        self.saveFreq( mag, label, Tag )
    
    def timeTxt( self, timeSize, family = 'db6' ):
#       batch saving de-noised EMG data in time domain
        ew = emgWave()
        ew.saveTxt( timeSize, family )
        
'''
class featureExtract - Generate feature matrix
extracting features in time domain, frequency domain and time-frequency domain from emg data

Input:
excelFilename - the name of output excel file which save feature matrix -> excel file

Output:
excel file - saving feature matrix -> file
features - EMG feature matrix -> matrix
Time Features:
MAV - Mean Absolute Value, the strength of EMG signal, MAV = 1/n * sum_(1~n)( |x| ) -> float
IEMG - integrated EMG, discharge energy of the exercise unit, IEMG= sum_(1~n)( |x| ) -> float
RMS - Root Mean Square, effective value of the exercise unit discharge, RMS = sqrt( 1/n * sum_(1~n)( x^2 ) ) -> float
WL - Waveform Length, WL = sum_(2~n)( | x[i]-x[i-1] | ) -> float
MAVS - Mean Absolute Value Slope, the tilt of emg curve, MAVS = 1/n * sum_(1~n)( | y[i]-y[i-1] | / | x[i]-x[i-1]| ) -> float
SSC - Slope Sign Changes, another measure of frequency content - the number of times the slope changes sign -> int
ZC - Zero Crossing, the number of times the waveform crosses zeros -> int
Frequency Features:
MNF - Mean Frequency, MNF = sum_(1~m)( fx * Py ) / sum_(1~m)( Py ) -> float
MDF - Median Frequency -> int
PkF - the peak of the amplitude of frequency domain, PkF = max( Py ) -> float
MNP - Mean Power, MNP = sum_(1~m)( Py )/m -> float
SMi - spectral moment, SM = sum_(1~m)( (fx**i) * py ) -> float
numPkF - the number of the peak in frequency domain -> int

Functions
featureMatrix - sequential batch processing EMG data in time and frequency domain, generate feature matrix and save them into excel file
readEmg - read EMG data ( time domain or frequency domain )
timeFeatures - create time domain features
MAVSlope, zeroCrossing, slopeSignChange - some of EMG features in time domain
frequencyFeatures - create frequency domain features
medianFrequency, numPeakFrequency - some of EMG features in frequency domain
'''

class featureExtract:
    def __init__():
        return
    
    def featureMatrix( self, excelFilename ):
#       sequential batch processing EMG data in time and frequency domain, generate feature matrix and save them into excel file
#       Input:
#       excelFilename - the name of output excel file which save feature matrix
#       Output:
#       EMG feature matrix and saving in excel file
        self.features = []
        for i in range( 1, 7 ):
            folderName = str(i)
            l = os.listdir( folderName + '/' )
            for name in l:
                fileName = folderName + '/' + name
                if fileName[-5] == 'f':
                    f, fmag, _ = self.readEmg( fileName )
                    MNF, MDF, PkF, MNP, SM, numPkF = self.frequencyFeatures( f, fmag ) 
                if fileName[-5] == 't':
                    t, mag, label = self.readEmg( fileName )
                    MAV, IEMG, RMS, WL, MAVS, SSC, ZC = self.timeFeatures( t, mag )
                    tmp = []
                    for i in range( 8 ):
                        tmp.extend( [ MAV[i], IEMG[i], RMS[i], WL[i], MAVS[i], SSC[i], ZC[i], \
                                     MNF[i], MDF[i], PkF[i], MNP[i], SM[i], numPkF[i] ] )
                    tmp.extend( [ label[0] ] )
                    self.features.append( tmp )
        fm = pd.DataFrame( self.features )
        fm.to_csv( excelFilename )
        return self.features
    
    def readEmg( self, filename ):
#       read EMG data ( time domain or frequency domain )
        fr = open( filename )
        lines = fr.readlines()
        t, mag, label = [], [], []
        for line in lines:
            tmp = line.split()
            t.append( float( tmp[0] ) )
            mag.append( tmp[1:-1] )
            label.append( int(tmp[-1]) )
        return t, mag, label
    
#   extract features from time domain
    def timeFeatures( self, t, mag ):
        self.MAV, self.IEMG, self.RMS, self.WL, self.MAVS, self.SSC, self.ZC = [], [], [], [], [], [], []
        for i in range( 8 ):
            emgVals = [ float( tmp[i] ) for tmp in mag ]
            emgAbsVals = [ abs( float( tmp[i] ) ) for tmp in mag ]
            N = len(mag)
            self.MAV.append( sum( emgAbsVals )/N )
            self.IEMG.append( sum( emgVals ) )
            self.RMS.append( float( np.sqrt( 1/N * sum( np.mat( emgVals ) * np.mat( emgVals ).T ) ) ) )
            self.WL.append( float( sum( [ abs( emgVals[i]-emgVals[i-1] ) for i in range( 1, len(emgVals) ) ]  ) ) )
            self.MAVS.append( self.MAVSlope( t, emgVals ) )
            self.SSC.append( self.slopeSignChange( emgVals ) )
            self.ZC.append( self.zeroCrossing( emgVals ) )
        return self.MAV, self.IEMG, self.RMS, self.WL, self.MAVS, self.SSC, self.ZC
    
    def MAVSlope( self, x, y ):
#       the average slope of Mean Absolute Value
        slopes = []
        for i in range( 1, len( y ) ):
            xt, yt = x[i] - x[i-1], abs( y[i] - y[i-1] )
            slopes.append( yt/xt )
        res = sum(slopes)/len(slopes)
        return res
    
    def zeroCrossing( self, y ):
#       # of times the curve crosses the X-axis
        res = 0
        threshold = np.std( y ) * 0.001 # threshold
        N = len( y )
        for i in range( 1, N ):
            sign = y[i] * y[i-1]
            val = abs( y[i] - y[i-1] )
            if sign < 0 and val > threshold:
                res += 1
        return res
    
    def slopeSignChange( self, y ):
#       # of times the slope changes sign
        res = 0
        threshold = np.std( y ) * 0.001 # threshold
        N = len( y )
        val = [ y[i]-y[i-1] for i in range( 1, N ) if y[i]-y[i-1] != 0 ]
        M = len(val)
        for i in range( 1, M ):
            if val[i-1]*val[i] < 0 and abs( val[i-1] ) > threshold and abs ( val[i] ) > threshold:
                res += 1
        return res
    
#   extract frequency domain
    def frequencyFeatures( self, f, fmag ):
        self.MNF, self.MDF, self.PkF, self.MNP, self.SM, self.numPkF = [], [], [], [], [], []
        for i in range( 8 ):
            fvals = [ float( tmp[i] ) for tmp in fmag ]
            m = len( fvals )

            self.MNF.append( float( sum( np.mat( fvals ) * np.mat(f).T ) / sum( fvals ) ) )
            self.MDF.append( self.medianFrequency( fvals ) )
            self.PkF.append( max( fvals ) )
            self.MNP.append( sum( fvals )/m )
            self.SM.append( float( sum( np.mat( fvals ) * np.mat(f).T ) ) )
            self.numPkF.append( self.numPeakFrequency( fvals ) )
        return self.MNF, self.MDF, self.PkF, self.MNP, self.SM, self.numPkF
    
    def medianFrequency( self, y ):
        n = 0
        ypre = sum( y[:1] )
        ypost = sum( y[1:])
        while ypre < ypost:
            n += 1
            ypre += y[n]
            ypost -= y[n] 
        if abs( sum( y[n:] ) - sum( y[:n] ) ) > abs( ypre - ypost ):
            return n
        else:
            return n-1
    
    def numPeakFrequency( self, y ):
#       # of peak in Frequency domain
        res = 0
        threshold = np.std( y ) * 0.5 # threshold
        for i in range( 1, len(y)-1 ):
            if y[i] > y[i-1] and y[i] > y[i+1] and ( y[i] - y[i-1] ) > threshold and ( y[i] - y[i+1] ) > threshold:
                res += 1
        return res

'''
class MLs - Machine Learning methods, for verifying method validity
Gaussian Naive Bayes, Decision Tree, Logistic Regression, Linear Discriminant Analysis
Stochastic Gradient Decent, Random Forest, Gradient Boosting, KNN and SVM

Inputs:
data - feature matrix -> matrix
N - # of testing samples -> int

Functions:
divideData - divide data into traing data and testing data
autoNorm - normalization
RocCurve - calculate fpr, tpr, roc_auc of ROC cure from test and score
RocPlot - draw ROC curve of all methods

Methods:
divideData - divide the data into N training data and N' testing data
GNB - Gaussian Naive Bayes
desionTree - Decision Tree
logisticRegression - Logistic Regression, solver = 'liblinear', multi_class = 'auto' 
LDA - linear Discriminant Analysis
SGD - Stochastic Gradient Decent, loss = "modified_huber", penalty = "l1" 
randomForest - Random Forest, n_estimators = 100
gradientBoost - Gradient Bossting, n_estimators = 100
KNN - KNN, algorithm = 'ball_tree', n_neighbors = 10
SVM - Support Vector Machine, linear SVM, random_state = 0
'''

import random as rd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# data = ( pd.read_csv( 'feature matrix.csv' ) ).values

class MLs:
    def __init__():
        return
    
    def divideData( self, data, N ):
        index1 = sorted( rd.sample( range( len( data ) ), N ) )
        index2 = [ x for x in range( len( data ) ) if x not in index1 ]
        data1 = data[index1]
        data2 = data[index2]
        trainData, trainLabel, testData, testLabel = [], [], [], []
        for tmp in data1:
            trainData.append( tmp[1:-1] )
            trainLabel.append( tmp[-1] )
        for tmp in data2:
            testData.append( tmp[1:-1] )
            testLabel.append( tmp[-1] )
        return trainData, trainLabel, testData, testLabel
    
    def autoNorm( self, dataSet ):
        label = [ x[-1] for x in dataSet ]
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        ranges = maxVals - minVals
        normDataSet = np.zeros(np.shape(dataSet))
        m = dataSet.shape[0]
        normDataSet = ( dataSet - np.tile(minVals, (m,1)) ) /np.tile(ranges, (m,1))   #element wise divide
        normDataSet = np.delete( normDataSet, -1, axis = 1 )
        normDataSet = np.append( normDataSet, np.array( label ).reshape( ( len(label) ), 1 ), axis = 1 )
        return normDataSet, ranges, minVals
    
    def RocCurve( self, test, score ):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        N = len( test[0] )
        for i in range( N ):
            fpr[i], tpr[i], _ = roc_curve( test[:, i], score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve( test.ravel(), score.ravel() )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        return fpr, tpr, roc_auc
    
    def RocPlot( self, data, N ):
        _, fprGnb, tprGnb, aucGnb = self.classifier( data, N, 'GNB' )
        _, fprDt, tprDt, aucDt = self.classifier( data, N, 'decisionTree' )
        _, fprLr, tprLr, aucLr = self.classifier( data, N, 'logisticRegression' )
        _, fprLda, tprLda, aucLda = self.classifier( data, N, 'LDA' )
        _, fprSgd, tprSgd, aucSgd = self.classifier( data, N, 'SGD' )
        _, fprRf, tprRf, aucRf = self.classifier( data, N, 'randomForest' )
        _, fprGb, tprGb, aucGb = self.classifier( data, N, 'gradientBoost' )
        _, fprKnn, tprKnn, aucKnn = self.classifier( data, N, 'kNN' )
        _, fprSvm, tprSvm, aucSvm = self.classifier( data, N, 'SVM' )
        methods = [ 'Gnb', 'Dt', 'Lr', 'Lda', 'Sgd', 'Rf', 'Gb', 'Knn', 'Svm' ]
        for m in methods: # eval - transfer srt into variable
            x, y, auc = eval( 'fpr'+m )['micro'], eval( 'tpr'+m )['micro'], eval( 'auc'+m )['micro']
            plt.plot( x, y, lw = 2, label = '{0:s} (area = {1:0.4f})'.format( m, auc) )
        plt.plot( [0, 1], [0, 1], 'k--', lw = 2 )
        plt.xlim( [-0.05, 1.0] )
        plt.ylim( [0.0, 1.05] )
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Micro-average ROC curve of Multi-class of different algorithms')
        plt.legend( loc  = 'lower right' )
        plt.show()
    
    def classifier( self, data, N, method ):
        if method == 'GNB':
            clf = GaussianNB()
        elif method == 'decisionTree':
            clf = DecisionTreeClassifier()
        elif method == 'lr':
            clf = LogisticRegression( solver = 'liblinear', multi_class = 'auto' )
        elif method == 'LDA':
            clf = LinearDiscriminantAnalysis()
        elif method == 'SGD':
            clf = SGDClassifier( loss = "modified_huber", penalty = "l1" )
        elif method == 'randomForest':
            clf = RandomForestClassifier( n_estimators = 100 )
        elif method == 'gradientBoost':
            clf = GradientBoostingClassifier( n_estimators = 100 )
        elif method == 'kNN':
            data, _, _ = self.autoNorm( data )
            clf = KNeighborsClassifier( algorithm = 'ball_tree', n_neighbors = 10 )
        elif method == 'SVM':
            data, _, _ = self.autoNorm( data )
            clf = LinearSVC( random_state = 0 )
        trainData, trainLabel, testData, testLabel = self.divideData( data, N )
        output = clf.fit( trainData, trainLabel ).predict( testData )
        n, Num = 0, len( testData )
        for i in range( Num ):
            if output[i] == testLabel[i]:
                n += 1
        accuracy = n / Num
        yscore = clf.fit( trainData, trainLabel ).predict_proba( testData )
        ytest = label_binarize( testLabel, classes=[ 1, 2, 3, 4, 5, 6 ] )
        fpr, tpr, roc_auc = self.RocCurve( ytest, yscore )
        return accuracy, fpr, tpr, roc_auc