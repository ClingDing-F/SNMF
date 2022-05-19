from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
import numpy as np

def generate_2_class_data(data_num, dim, bias):
    mean1 = np.random.rand(512) + 0.2
    X1, y1 = make_gaussian_quantiles(mean=mean1, cov=2,
                                     n_samples=data_num, n_features=dim,
                                     n_classes=1, random_state=1)

    # Construct dataset
    X2, y2 = make_gaussian_quantiles(mean=2+ mean1 + bias, cov=3.,
                                     n_samples=data_num, n_features=dim,
                                     n_classes=1, random_state=0)

    y2 = np.asarray([1] * data_num)

    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    # X=np.where(X >= 0, X, 0)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.259)

    return X_train, X_test, Y_train, Y_test