# EMG-classification
Classification of Gestures from Electromyography Signal  
It introduced a methodology to make the classification of gestures from EMG signals:  
First: the pre-processing method for de-noising and eliminating possible artifacts via wavelet function,  
Second: the EMG signals are extracted to construct a feature matrix, and machine learning methods are used to classify the gestures from the data,  
Finally, compares various algorithms in terms of classification performance to explore the best classifiers for the classification and prediction of the gestures from the EMG data.

(Flowchart of methodology via wavelet function)
![Image text](https://github.com/arialibra/EMG-classification/blob/master/IMG-folder/flowchart.jpg)

After compared the accuracies and ROC curve of the result, except for the Decision Tree classier, all other algorithms perform well and the Random Forest classier acquires the highest accuracy and AUC value which is the best classication model among all the algorithms.

![Image text](https://github.com/arialibra/EMG-classification/blob/master/IMG-folder/accandROC.jpg)
