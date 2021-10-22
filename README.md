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
This function takes in the data X (ndarray; trials by channels by time), labels y (ndarray; vector), and a time (ndarray, vector).

```Python
from temp_dec import decoding_functions
decoding_functions.temporal_decoding(X,y,time)
```



#### Using a sliding time window
If there is information in the temporal dynamics of the signal, using a sliding time window will increase decoding accuracy (and smooth the signal). We also demean the signal within each window, this avoids the issue of baselining. 
```Python
temporal_dymanics = True
size_window=5
```


#### Applying PCA
If you use a large amount of features, you might want to consider applying PCA to your features before applying your classifier. In addition, classifiers are sensitive to noise rejecting noise components from the data can be beneficial. 

```Python
use_pca = True
```
You can also regulate how many components you would like to keep (setting the pca_components variant to > 1) or how much variance you would like to explain (setting the pca_components variant to < 1). As a general rule of thumb maintaining 95% of variance will maintain enough signal and reduces feature space. 

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

#### All options incorporated


``` Python
output = decoding_functions.temporal_decoding(data,labels,time,
                                                n_bins=number_of_categories,
                                                n_folds=10,
                                                classifier='LDA',
                                                use_pca=True,pca_components=.95,
                                                temporal_dynamics=True,
                                                size_window=20)
```
