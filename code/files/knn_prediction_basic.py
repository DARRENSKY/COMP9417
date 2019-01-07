from scipy.io import arff
from statistics import mean
import numpy as np
import pandas as pd
# idea: get k neighbours out of n training examples, get the most frequent class

def dimension_reduction(data, metadata):
    names = list()
    for index, type in enumerate(metadata.types()):
        if type =='nominal':
            # drop nominal
            names.append(metadata.names()[index])

    data = data.drop(names, axis=1)
    data = data.dropna(axis=0)
    data = data.reset_index()
    return data

def euclidean_distance(training_data, inX):
    dif = np.tile(inX, (training_data.shape[0], 1)) - training_data
    sqaured_diff = dif.applymap(np.square)
    # column sum
    sum_diff = sqaured_diff.sum(axis=1)
    return sum_diff.map(np.sqrt)

def knn_regression(training_data, inX, labels, k):
    assert k<=len(training_data)
    #distance
    distance = euclidean_distance(training_data, inX)
    # sort distance in ascending order, remove None
    sorted_index = distance.argsort()
    # get the top k rows from the sorted array
    top_k_labels = labels[sorted_index[:k]]
    return sum(top_k_labels)/len(top_k_labels)

def cross_validation_knn_prediction(k):
    dataset_prediction = arff.loadarff('autos.arff')
    # this is a table with 350 rows and 34 features, and last dimension as class
    training_data_prediction = pd.DataFrame(dataset_prediction[0])
    training_data_prediction = dimension_reduction(training_data_prediction, dataset_prediction[1])
    training_data_prediction = training_data_prediction.reset_index(drop=True)
    MAPE = []
    for i in range(training_data_prediction.shape[0]):
        training_data_inc_class = pd.concat([training_data_prediction.iloc[:i, :], training_data_prediction.iloc[i+1:, :]], ignore_index=True)
        labels = training_data_inc_class['price']
        training_data = training_data_inc_class.iloc[:,:-1]
        inX = training_data_prediction.iloc[i:i+1, :-1]
        # print("-------------------")
        est_price = knn_regression(training_data, inX, labels, k)
        act_price = training_data_prediction['price'][i]
        act_est = abs((act_price - est_price) / (act_price))
        # print("Deviation Error of Prediction from Actual:")
        # print(act_est)
        MAPE.append(act_est)
    mape = mean(MAPE)
    print('k = %d' % k)
    print("Results: Percent Average Deviation Error of Prediction from Actual: ", mape)
    return mape

if __name__ == '__main__':
    k = 10
    cross_validation_knn_prediction(k)
