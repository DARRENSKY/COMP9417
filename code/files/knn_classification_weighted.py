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


def weight(distance):
    return 1.0/(distance+0.001)**2

# knn method based on pandas data operations
def knn_classification_weighted(training_data, inX, labels, k):
    assert k <= training_data.shape[1]
    #distance
    distance = euclidean_distance(training_data, inX)
    # sort distance in ascending order, remove None
    sorted_index = distance.argsort()
    top_k_labels = labels[sorted_index[:k]]
    top_k_distance = distance[sorted_index[:k]]
    top_k_weights = top_k_distance.tolist()
    # ！！！有些weights可能是0， when i=102
    # 暂时+微小系数来处理
    top_k_weights = [weight(feature) for feature in top_k_weights]
    rankDict = dict()
    for index in range(top_k_labels.shape[0]):
        if top_k_labels.iloc[index] not in rankDict.keys():
            rankDict[top_k_labels.iloc[index]]=0
        rankDict[top_k_labels.iloc[index]]+=top_k_weights[index]
    max = 0
    max_label = None
    for key in rankDict.keys():
        if max < rankDict[key]:
            max = rankDict[key]
            max_label = key
    return max_label

def cross_validation_wnn_classfication(k):
    count_success=0
    count_execute=0
    dataset_classification = arff.loadarff('ionosphere.arff')
    # this is a table with 350 rows and 34 features, and last dimension as class
    training_data_classification = pd.DataFrame(dataset_classification[0])

    for i in range(training_data_classification.shape[0]):
        count_execute+=1
        training_data = pd.concat([training_data_classification.iloc[:i, :], training_data_classification.iloc[i+1:, :]], ignore_index=True)
        labels = training_data['class']
        training_data = training_data.iloc[:,:-1]
        inX = training_data_classification.iloc[i:i+1, :-1]
        # print('i = %d'%i)
        # print(training_data_classification.iloc[i:i+1,:]['class'][0]) bytes

        if(training_data_classification.iloc[i:i+1, :]['class'].tolist()[0] == knn_classification_weighted(training_data, inX, labels, k)):
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
    k = 10
    cross_validation_wnn_classfication(k)
