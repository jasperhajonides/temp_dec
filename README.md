# Supervised Learning Function for Neural data

This toolbox facilitates neural decoding of time series. Under the hood it uses scikit-learn functions.

## Set-up
Install using pip (https://pypi.org/project/temp-dec/)

```unix
pip install temp_dec
```


## Requirements

```Python
sklearn==0.21.3
numpy==1.13.1
scipy==0.19.1
```

## Python implementation
This function takes in the data X (ndarray; trials by features by time), labels y (ndarray; vector).

```Python
from temp_dec import decoding_functions
decoding_functions.temporal_decoding(X, y)
```



#### Using a sliding time window
If there is information in the temporal dynamics of the signal, using a sliding time window will increase decoding accuracy (and smooth the signal). We can also demean the signal within each window, this avoids the issue of baselining. 

```Python
size_window=5
demean=True
```


#### Applying PCA
If you use a large amount of features, you might want to consider applying PCA to your features before applying your classifier. In addition, classifiers are sensitive to noise rejecting noise components from the data can be beneficial. 
You can also regulate how many components you would like to keep (setting the pca_components variant to > 1) or how much variance you would like to explain (setting the pca_components variant to < 1). As a general rule of thumb maintaining 95% of variance will maintain enough signal and reduces feature space. If `pca_components = 1` then 100% of the variance will be maintained so no PCA is applied.

```Python
pca_components = .95
```


#### Classifiers
Different classifiers are supported, selected in accordance with Grootwagers et al (2017) j.cogn.neurosci.
* LDA: linear disciminant analysis
* LG: logistic regression
* GNB: Gaussian Naive Bayes
* maha: Nearest Neighbours using mahalanobis distance. 


```Python
classifier = 'LDA'
```

with the amounts of stratified cross-validations (kfold) adjusted with the following flag, 5-fold by default. 

```Python
n_folds = 5
```

#### All options incorporated


``` Python
output = decoding_functions.temporal_decoding(data,labels,
                                                n_folds=5,
                                                classifier='LDA',
                                                pca_components=.95,
                                                size_window=20,
                                                demean=True)
```
