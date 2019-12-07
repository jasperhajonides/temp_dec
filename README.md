# Supervised Learning Toolbox for Neural data


This toolbox facilitates neural decoding of time series. Under the hood it uses mainly scikit-learn functions.
* Single and multi time point decoding 
* Dissimilarity analyses (yet to be added)

Download these functions and add them by running:

```Python
import sys
sys.path.append('Functions')
from decoding_functions import *
```



## Single and multi time point decoding 
This function takes in the data X (ndarray; trials by channels by time), labels y (ndarray; vector), and a time (ndarray, vector).



#### Using a sliding time window
If there is information in the temporal dynamics of the signal, using a sliding time window will increase decoding accuracy (and smooth the signal). We also demean the signal within each window, this avoids the issue of baselining. 
```Python
temporal_dymanics == True
```
#### Applying PCA
If you use a large amount of features, you might want to consider applying PCA to your features before applying your classifier. In addition, LDA is fairly sensitive to noise to in terms of denoising this could also be beneficial. 

```Python
use_pca == True
```
You can also regulate how many components you would like to keep or how much variance you would like to explain. As a general rule of thumb maintaining 95% of variance will maintain enough signal and reduces feature space.

```Python
pca_components == .95
```


[work in progress.]
