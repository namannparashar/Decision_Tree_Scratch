# SCC 461 Programming For Data Scientists

import numpy as np 
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None): 
        
        # decision node 
        self.feature_index = feature_index 
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # leaf node
        self.value = value

class DecisionTreeClassifier_sc():
    def __init__(self, min_samples_split=2, max_depth=2, criterion = "gini"):
        
        # initializing the root of the tree 
        self.root = None
        
        # splitting criterions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion 
        
    def fit(self, X, Y):
        # fitting the data
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
        
    def build_tree(self, dataset, curr_depth=0):
        # building the tree recursively
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # running the loop until the splitting criterions are met
        if num_samples>=self.min_samples_split and curr_depth < self.max_depth:
            # getting the values corresponding to best split 
            feature_index_best, threshold_best, dataset_left_best, dataset_right_best, info_gain_best = self.get_best_split(dataset, num_samples, num_features)
            # building tree if info_gain is +ve 
            if info_gain_best >0:
                # left branch
                left_branch = self.build_tree(dataset_left_best, curr_depth+1)
                # right branch
                right_branch = self.build_tree(dataset_right_best, curr_depth+1)
                # decision node
                return Node(feature_index_best, threshold_best, 
                            left_branch, right_branch, info_gain_best) 
        
        # computing leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value) 

    def get_best_split(self, dataset, num_samples, num_features):
      # getting the best split of the dataset
        
        feature_index_best = None
        threshold_best = None 
        dataset_left_best = None 
        dataset_right_best = None 
        info_gain_best = None  
        max_info_gain = -float("inf")
        
        for feature_index in range(num_features):
          # looping through all the features
            feature_values = dataset[:, feature_index]
            # getting the unique values of a feature
            features_unique = np.unique(feature_values)
            features_sorted = np.sort(features_unique)
            # getting the mid-points of the values of the feature
            for j in range(len(features_sorted) - 1):
              threshold = (features_sorted[j] + features_sorted[j+1])/2
            # splitting the dataset into left and right based on the mid-points
              dataset_left, dataset_right = dataset[dataset[:, feature_index] < threshold], dataset[dataset[:, feature_index] >= threshold] 
              if len(dataset_left)>0 and len(dataset_right)>0:
                # getting the information gain post data split
                  curr_info_gain = self.information_gain(dataset, dataset_left, dataset_right, self.criterion)
                  # updating the values of decision variables for the best split 
                  if curr_info_gain>max_info_gain:
                      feature_index_best = feature_index
                      threshold_best = threshold
                      dataset_left_best = dataset_left
                      dataset_right_best = dataset_right
                      info_gain_best = curr_info_gain 
                      max_info_gain = curr_info_gain
        return feature_index_best, threshold_best, dataset_left_best, dataset_right_best, info_gain_best

    def information_gain(self, parent, l_child, r_child, mode):
        # calculating information gain

        ## getting the weights
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        # getting information gain value
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self,y):
      # calculating entropy value for splits
      y_samples, y_features = np.shape(y)
      unique_labels = list(set(y[:,-1]))
      e = 0 
      for attributes in unique_labels:
        l = [row[-1] for row in y].count(attributes)/y_samples   
        e += -l * np.log2(l)
      return e 
    
    def gini_index(self,y):
      # caculating gini value for splits
      y_samples, y_features = np.shape(y)
      unique_labels = list(set(y[:,-1]))
      p = 0 
      for attributes in unique_labels:
        l = [row[-1] for row in y].count(attributes)/y_samples   
        p += l*l 
      p = (1 - p)
      return p

    def calculate_leaf_value(self, Y):
        # getting most frequent target value
        unique, indices = np.unique(Y, return_counts=True)
        return unique[np.argmax(indices)]

    
    def make_prediction(self, x, tree):
        # calculating values for prediction
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
            
    def predict(self, X):
        # predicting the values
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions