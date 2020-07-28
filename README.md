# Supervised Learning Toolbox for Neural data

This toolbox facilitates neural decoding of time series. Under the hood it uses scikit-learn functions.

## Set-up
Download these functions and add them by running:

```Python
cd /path/to/directory
git clone https://github.com/jasperhajonides/Supervised_Learning_TB.git
```

Now add this path to your Python settings:

```Python
export PYTHONPATH=/path/to/directory/Supervised_Learning_TB:$PYTHONPATH
```


## Requirements

```Python
matplotlib==2.0.2
numpy==1.13.1
scipy==0.19.1
```

## Example
This function takes in the data X (ndarray; trials by channels by time), labels y (ndarray; vector), and a time (ndarray, vector).

```Python
import Supervised_Learning_TB
```



#### Using a sliding time window
If there is information in the temporal dynamics of the signal, using a sliding time window will increase decoding accuracy (and smooth the signal). We also demean the signal within each window, this avoids the issue of baselining. 
```Python
temporal_dymanics == True
```


#### Applying PCA
If you use a large amount of features, you might want to consider applying PCA to your features before applying your classifier. In addition, classifiers are sensitive to noise rejecting noise components from the data can be beneficial. 

```Python
use_pca == True
```
You can also regulate how many components you would like to keep (setting the pca_components variant to > 1) or how much variance you would like to explain (setting the pca_components variant to < 1). As a general rule of thumb maintaining 95% of variance will maintain enough signal and reduces feature space. 

```Python
pca_components == .95
```



#### Classifiers
Different classifiers are supported, selected in accordance with Grootwagers et al (2017) j.cogn.neurosci.
* LDA: linear disciminant analysis
* LG: logistic regression
* GNB: Gaussian Naive Bayes
* maha: Nearest Neighbours using mahalanobis distance. 


```Python
classifier == 'LDA'
```

