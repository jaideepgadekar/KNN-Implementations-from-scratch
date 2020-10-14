import numpy as np
import csv
import sys
def import_data():
    X = np.genfromtxt("train_X_knn.csv", delimiter = ",", \
                      dtype = np.float64, skip_header = 1)
    Y = np.genfromtxt("train_Y_knn.csv", delimiter =  ",", \
                      dtype = np.float64)
   
    return X, Y
def compute_ln_norm_distance(vector1, vector2, n):
    vector_len = len(vector1)
    diff_vector = []

    for i in range(0, vector_len):
      abs_diff = abs(vector1[i] - vector2[i])
      diff_vector.append(abs_diff ** n)
    ln_norm_distance = (sum(diff_vector))**(1.0/n)
    return ln_norm_distance

def compute_hamming_distance(vector1, vector2):
    vector_len = len(vector1)
    hamming_distance = 0

    for i in range(0, vector_len):
      if(vector1[i] != vector2[i]):
        hamming_distance += 1 

    return hamming_distance


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
def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = np.array(pred_Y)
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline = '') as file:
        wr = csv.writer(file)
        wr.writerows(pred_Y)
        file.close()

def calculate_accuracy(predicted_Y, actual_Y):
    total_num_of_observations = len(predicted_Y)
    num_of_values_matched = 0
    for i in range(0,total_num_of_observations):
        if(predicted_Y[i] == actual_Y[i]):
            num_of_values_matched +=1
    return float(num_of_values_matched)/total_num_of_observations

def predict(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter = ",", \
                      dtype = np.float64, skip_header = 1)
    pred_Y = classify_points_using_knn(X, Y, test_X, 2, 2)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")

if __name__ == "__main__":
    X, Y =import_data()
    predict(sys.argv[1])
     
