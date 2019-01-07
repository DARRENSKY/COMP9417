from scipy.io import arff
import numpy as np
import pandas as pd
# idea: get k neighbours out of n training examples, get the most frequent class


def euclidean_distance(training_data, inX):
    dif = np.tile(inX, (training_data.shape[0], 1)) - training_data
    sqaured_diff = dif.applymap(np.square)
    # column sum
    sum_diff = sqaured_diff.sum(axis=1)
    return sum_diff.map(np.sqrt)

#original knn
def knn_classification(training_data, inX, labels, k):
    assert k<=len(training_data)
    #distance
    distance = euclidean_distance(training_data, inX)
    # sort distance in ascending order, remove None
    sorted_index = distance.argsort()
    # get the top k rows from the sorted array

    top_k_labels = labels[sorted_index[:k]]
    # get the most frequent class of these rows, as predicted class
    label_sort = top_k_labels.value_counts()
    return label_sort.index[0]

def cross_validation_knn_classfication(k):
    count_success=0
    count_execute=0
    dataset_classification = arff.loadarff('ionosphere.arff')
    # this is a table with 351 rows and 34 features, and last dimension as class
    training_data_classification = pd.DataFrame(dataset_classification[0])

    for i in range(training_data_classification.shape[0]):
        count_execute+=1
        training_data = pd.concat([training_data_classification.iloc[:i, :], training_data_classification.iloc[i+1:, :]], ignore_index=True)
        labels = training_data['class']
        training_data = training_data.iloc[:,:-1]
        inX = training_data_classification.iloc[i:i+1, :-1]
        if(training_data_classification.iloc[i:i+1, :]['class'].tolist()[0] == knn_classification(training_data, inX, labels, k)):
            # print('true.')
            count_success+=1
        else:
            pass
            # print('false')
        # print('----------------------------------')
    print('k = %d'%k)
    print('success rate: %f'%(count_success/count_execute))
    return count_success/count_execute

if __name__ == '__main__':
    k=10
    cross_validation_knn_classfication(k)
