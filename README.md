# KNN-Implementations-from-scratch

In this repo, the task was to identify type of plastics found on the sea to reduce the use of that plastic. We have to find type of plastic using information about its remains. For this classification task I have used knn-algorithm implemented from scratch in python and using numpy.

- **train_X, train_Y** - training data

- Atrributes of train_x = %chlorine, %hydrogen, hardness, %sulphur, %carbon. %oxygen, %nytrogen.
- labels in train-Y = 1,2,3,4,5,6

## Code

```python3
#loading necessary libraries
import numpy as np
import csv
import sys
```
```python3
#loading data 
def import_data():
    X = np.genfromtxt("train_X_knn.csv", delimiter = ",", \
                      dtype = np.float64, skip_header = 1)
    Y = np.genfromtxt("train_Y_knn.csv", delimiter =  ",", \
                      dtype = np.float64)
   
    return X, Y
```
```python3
#computing L-norm of the vectors(only valid for numerical data) 
def compute_ln_norm_distance(vector1, vector2, n):
    vector_len = len(vector1)
    diff_vector = []

    for i in range(0, vector_len):
      abs_diff = abs(vector1[i] - vector2[i])
      diff_vector.append(abs_diff ** n)
    ln_norm_distance = (sum(diff_vector))**(1.0/n)
    return ln_norm_distance
```
```python3
#finding K Nearest neigbour of the test Example
def find_k_nearest_neighbors(train_X, test_example, k, n_in_ln_norm_distance):
    indices_dist_pairs = []
    index= 0
    for train_elem_x in train_X:
      distance = compute_ln_norm_distance(train_elem_x, test_example,n_in_ln_norm_distance)
      indices_dist_pairs.append([index, distance])
      index += 1
    indices_dist_pairs.sort(key = lambda x: x[1])
    top_k_pairs = indices_dist_pairs[0:k]
    top_k_indices = [i[0] for i in top_k_pairs]
    return top_k_indices
```
```python3
#Labelling the data in test_x using its Labels of K nearest neighbours
def classify_points_using_knn(train_X, train_Y, test_X, n_in_ln_norm_distance, k):
    test_Y = []
    for test_elem_x in test_X:
      top_k_nn_indices = find_k_nearest_neighbors(train_X, test_elem_x, k,n_in_ln_norm_distance)
      top_knn_labels = []
      for i in top_k_nn_indices:
        top_knn_labels.append(train_Y[i])
      most_frequent_label = max(set(top_knn_labels), key = top_knn_labels.count)
      test_Y.append(most_frequent_label)
    return test_Y
```
```python3
#Saving predicted labels to csv file
def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = np.array(pred_Y)
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline = '') as file:
        wr = csv.writer(file)
        wr.writerows(pred_Y)
        file.close()
```
```python3
#Calculating Accuracy of our prediction
def calculate_accuracy(predicted_Y, actual_Y):
    total_num_of_observations = len(predicted_Y)
    num_of_values_matched = 0
    for i in range(0,total_num_of_observations):
        if(predicted_Y[i] == actual_Y[i]):
            num_of_values_matched +=1
    return float(num_of_values_matched)/total_num_of_observations
```
```python3
#predicting for the test data
def predict(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter = ",", \
                      dtype = np.float64, skip_header = 1)
    pred_Y = classify_points_using_knn(X, Y, test_X, 2, 2)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")
```
```python3

if __name__ == "__main__":
    X, Y =import_data()
    predict(sys.argv[1])
```
