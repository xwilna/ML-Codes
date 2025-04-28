from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

X = np.array([1,0, np.nan, 1, 1]).reshape(-1,1)
print(imputer.fit_transform(X))