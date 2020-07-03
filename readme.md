Linear multi-class classification SVM with no kernel
====================================================

Implemented three strategies of multi-class classification using linear SVM.

One vs Rest:
------------

### Accuracy

Achieved average accuracy of 73% when 5 fold cross validation is performed.

### Mapping function

Linear Mapping

### Loss function

The loss function is *Squared Hinge Loss*,

\(l(w,b,(x,y)) = ( \max \hspace{3pt} \{ 0, 1 -y( w \cdot (x) + b) \} )^2 \)

Crammer - Singer’s method
-------------------------

### Accuracy

Achieved average accuracy of 77% when 5 fold cross validation is performed.

### Mapping function

The mapping function is based on *Multivector Construction*, we deﬁne \(\Psi : X \times Y \rightarrow \mathbb{R}^d\), where \(d = nk\), \(Y = {1, \dots, 10} \) and \(X = \mathbb{R}^n\)

\(\Psi(x,y) = [ \hspace{3pt} \underbrace{0,\dots,0}_\textrm{ \)<sup>(y-1)n</sup>\( }, \hspace{3pt}
    \underbrace{x_1,\dots,x_n}_\textrm{ \)<sup>n</sup>\( },\hspace{3pt}
    \underbrace{0,\dots,0}_\textrm{ \)<sup>(k-y)n</sup>\( }
    \hspace{3pt} ]\)

### Loss function

The loss function is *Generalized Hinge Loss*,

\(l(w,(x,y)) = \max\limits_{y'\in Y} \hspace{3pt} (\Delta(y',y) + w . (\Psi(x,y') - \Psi(x,y)))\)

All pairs (one vs one):
-----------------------

### Accuracy

Achieved average accuracy of 85% when 5 fold cross validation is performed.

### Mapping function

Linear Mapping

### Loss function

The loss function is *Squared Hinge Loss*,

\(l(w,(x,y)) = ( \max \hspace{3pt} \{ 0, 1 -y( w \cdot (x)) \} )^2 \)

Estimating performance
======================

Empirical Methods used:
-----------------------

Performed 5-fold stratified cross validation and calculated the 0-1 loss and Hinge loss. 0-1 loss: \[\frac{1}{N}\sum_{i=1}^{N} \mathds{1}({y_i(w\cdot x_i) < 0 } )\] Hinge loss: \[\frac{1}{N}\sum_{i=1}^{N} (\max \hspace{3pt} \{ 0, 1 -y_i( w \cdot (x_i)) \} )\] where \(N\) is the total number of samples in the test dataset for a particular fold.

Since in class we were taught PAC bounds for binary classifiers based on VC dimensions. We take the example of the binary classifiers trained to follow one vs rest strategy. Consider the case of 0 vs Rest, The train 0-1 error is found to be \(training(x) = 0.017\) and test 0-1 error is found to be \(0.018\) on a train-test split of 80%, which gives a training set of size \(n=2997\). To calculate the upper bound of error we use PAC bound \#2 taught in class

\[\epsilon < \frac{1}{n}\left(training\left(X\right)\ +\ 4\ln\left(\frac{4}{d}\right)+\ 4\left(VC\left(h\right)\right)\ln\left(\frac{2e\cdot n}{VC\left(h\right)}\right)\right)\] here we assume \(d=0.05\) and we get the value \[\epsilon < \frac{1}{2997}\left(0.017\ +\ 4\ln\left(\frac{4}{0.05}\right)+\ 4\left(9\right)\ln\left(\frac{2e\cdot2997}{9}\right)\right)\] to give an upper bound \[\epsilon < 0.096\] which is acceptable since our test error came out to be \(0.018\). We can perform similar tests on the other cases and verify that this bound condition is always satisfied with the provided data.

Test set predictions
====================

Available in \(prashanth\_test\_preds.txt\), provided the predictions using crammer-singer’s method as it is comparable to most other students’ submissions.

Transfer learning with SVM
==========================

Performed transfer learning by taking 1 vs 9 as source problem and 1 vs 7 as target problem. Split the dataset with only 20% as training set and rest as test set to model the fact that transfer learning is most useful when there is less training data. The error estimate on the target problem without transfer learning is given in the table below.

[h]

|**Method**|**0-1 loss**|**Hinge Loss**|
|:---------|:-----------|:-------------|
|No Transfer|0.882|1.249|
|Hypothesis Transfer|0.883|1.256|
|Instance Transfer|0.787|1.964|

Hypothesis transfer seems to perform slightly better than no transfer which in turn performs better than instance transfer. This is most likely due to the weights of the target problem are calculated by taking some information about the weight vectors of the source problem and further minimizing the objective. This only captures information about the slope of the hyperplane but not the intercept. Whereas in the case of instance transfer since, we capture information about the data points itself, if the points in the source problem are not similar to the points in the target problem, the performance gets worse. In our data’s case, the way 9s are written is quite different from the way 7s are written, and this is the most likely cause of worse performance of instance transfer. Here is a plot of the distributions of the features in 9 (red) and 7 (blue), you can see that some features in 7 are quite different from 9.

![image](dist.png)

Bonus: Kernel SVM
=================

Used One vs rest strategy and trained kernel SVM on the dual formulation of the primal SVM problem with a Gaussian radial basis function (rbf) defined by \[K\left(x,x'\right) =\exp(-\gamma \|x-x'\|^2)\] where \(\gamma = 1 / (d \times variance(X))\) is a regularization parameter and \(d\) is the number of features. The new 0-1 error estimate is 0.076. The predictions for the test set are available in \(prashanth\_test\_kernel\_preds.txt\).
