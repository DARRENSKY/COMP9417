import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.mlab import PCA
from knn_classification_basic import cross_validation_knn_classfication,knn_classification
from knn_classification_weighted import cross_validation_wnn_classfication
from knn_prediction_basic import cross_validation_knn_prediction
from knn_prediction_weighted import cross_validation_wnn_prediction
from knn_prediction_exfeatures_basic import cross_validation_knn_prediction_extra_features
from knn_prediction_exfeatures_weighted import cross_validation_wnn_prediction_extra_features
from scipy.io import arff
import numpy as np
import pandas as pd
#Dimensionality reduction Plot
def plot_knn_high_dimensionality(k):
    dataset_classification = arff.loadarff('ionosphere.arff')
    # this is a table with 351 rows and 34 features, and last dimension as class
    training_data_classification = pd.DataFrame(dataset_classification[0])
    input_labels = training_data_classification['class']
    labels = training_data_classification['class'].str.decode("utf-8")
    training_data_pd = training_data_classification.iloc[:, :-1]
    training_data_np = training_data_pd.values
    labels_np = labels.values
    n_neighbors = k
    # PCA start--method
    training_data_np[training_data_np == 0] = 0.0001
    training_data_np.astype(float)
    pca = PCA(training_data_np)
    X = pca.Y[:, :2]
    # lab PCA start--method
    y = labels_np

    y[y == "b"] = 0
    y[y == "g"] = 1
    y = y.astype('int')
    h = .2
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

    # we create an instance of Neighbours Classifier and fit the data.

    # calculate min, max and limits
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    ready_test = np.c_[xx.ravel(), yy.ravel()]
    Z = []
    training_data_pd = pd.DataFrame(X)
    ready_test = pd.DataFrame(ready_test)
    for i in range(ready_test.shape[0]):
        pred = knn_classification(training_data=training_data_pd, inX=ready_test.iloc[i:i + 1, :], labels=input_labels,
                                  k=n_neighbors)
        if pred.decode("utf-8") == "b":
            Z.append(0)
        else:
            Z.append(1)
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2-Class classification (k = %i)" % (n_neighbors))
    plt.show()

def plot_knn_classfication():
    result = []
    for k in range(1, 101):
        if k % 2 != 0:
            success_rate = cross_validation_knn_classfication(k)
            result.append(1 - success_rate)
    ##plot the graph
    x_axies = []
    y_axies = []
    error_min = 999

    for i in range(1, len(result) + 1):
        x_axies.append(i)
        error_rate = result[i - 1]
        y_axies.append(error_rate)
        if error_rate < error_min:
            error_min = error_rate
            error_min_index = i
    print("The best K in KNN Classification: ", error_min_index)
    print("Error rate: ", error_min)
    plt.plot(x_axies, y_axies)
    plt.title("Basic k-NN Classification")
    plt.ylabel('Misclassification Error')
    plt.xlabel('Number of Neighbors k')
    plt.show()
def plot_wnn_classfication():
    result = []
    for k in range(1, 101):
        success_rate = cross_validation_wnn_classfication(k)
        result.append(1 - success_rate)
    ##plot the graph
    x_axies = []
    y_axies = []

    error_min = 999
    for i in range(1, len(result) + 1):
        x_axies.append(i)
        error_rate = result[i - 1]
        y_axies.append(error_rate)
        if error_rate < error_min:
            error_min = error_rate
            error_min_index = i

    print("The best K in WNN Classification: ", error_min_index)
    print("Error rate: ", error_min)
    plt.plot(x_axies, y_axies)
    plt.title("Distance weighted K-NN Classification")
    plt.ylabel('Misclassification Error')
    plt.xlabel('Number of Neighbors k')
    plt.show()

def plot_knn_prediction():
    result = []
    for k in range(1, 101):
        if k % 2 != 0:
            error_rate = cross_validation_knn_prediction(k)
            result.append(error_rate)
    ##plot the graph
    x_axies = []
    y_axies = []
    error_min = 999

    for i in range(1, len(result) + 1):
        x_axies.append(i)
        error_rate = result[i - 1]
        y_axies.append(error_rate)
        if error_rate < error_min:
            error_min = error_rate
            error_min_index = i
    print("The best K in Numerical Predication: ", error_min_index)
    print("Error rate: ", error_min)
    plt.plot(x_axies, y_axies)
    plt.title("Basic k-NN Numerical Predication")
    plt.ylabel('Percent Average Deviation Error of Prediction from Actual')
    plt.xlabel('Number of Neighbors k')
    plt.show()

def plot_wnn_prediction():
    result = []
    for k in range(1, 101):
        error_rate = cross_validation_wnn_prediction(k)
        result.append(error_rate)
    ##plot the graph
    x_axies = []
    y_axies = []

    error_min = 999
    for i in range(1, len(result) + 1):
        x_axies.append(i)
        error_rate = result[i - 1]
        y_axies.append(error_rate)
        if error_rate < error_min:
            error_min = error_rate
            error_min_index = i

    print("The best K in Numerical Prediction by KNN Regression Weighted: ", error_min_index)
    print("Error rate: ", error_min)
    plt.plot(x_axies, y_axies)
    plt.title("Distance weighted K-NN Numerical Predication")
    plt.ylabel('Percent Average Deviation Error of Prediction from Actual')
    plt.xlabel('Number of Neighbors k')
    plt.show()

def plot_knn_prediction_extra_features():
    result = []
    for k in range(1, 101):
        if k % 2 != 0:
            error_rate = cross_validation_knn_prediction_extra_features(k)
            result.append(error_rate)
    ##plot the graph
    x_axies = []
    y_axies = []
    index_y = 1
    error_min = 999
    for i in range(1, len(result) + 1):
        x_axies.append(index_y)
        error_value = result[i - 1]
        y_axies.append(error_value)
        if error_value < error_min:
            error_min = error_value
            error_index = index_y
        index_y += 2
    print("The best K in Numerical Prediction by KNN Regression(encoding extra features) is:", error_index)
    print("The Lowest Percent Average Deviation Error of Prediction from Actual:", error_min)
    plt.plot(x_axies, y_axies)
    plt.title("Basic K-NN Numerical Predication(encoding extra features)")
    plt.ylabel('Percent Average Deviation Error of Prediction from Actual')
    plt.xlabel('Number of Neighbors k')
    plt.show()

def plot_wnn_prediction_extra_features():
    result = []
    for k in range(1, 101):
        # if k % 2 != 0:
        error_rate = cross_validation_wnn_prediction_extra_features(k)
        result.append(error_rate)
    ##plot the graph
    x_axies = []
    y_axies = []

    error_min = 999
    for i in range(1, len(result) + 1):
        x_axies.append(i)
        error_value = result[i - 1]
        y_axies.append(error_value)
        if error_value < error_min:
            error_min = error_value
            error_index = i

    print("The best K in Numerical Prediction by WNN Regression(encoding extra features) is:", error_index)
    print("The Lowest Percent Average Deviation Error of Prediction from Actual:", error_min)
    plt.plot(x_axies, y_axies)
    plt.title("Distance weighted K-NN Numerical Predication(encoding extra features)")
    plt.ylabel('Percent Average Deviation Error of Prediction from Actual')
    plt.xlabel('Number of Neighbors k')
    plt.show()
if __name__ == '__main__':

    # plot_knn_high_dimensionality(k=1)

    # plot_knn_classfication()

    # plot_wnn_classfication()

    # plot_knn_prediction()

    # plot_wnn_prediction()

    # plot_knn_prediction_extra_features()

    plot_wnn_prediction_extra_features()
