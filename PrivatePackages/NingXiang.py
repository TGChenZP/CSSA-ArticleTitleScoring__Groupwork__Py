import pandas as pd
import numpy as np
import pickle
import copy

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression





class NingXiang:



    def __init__(self):
        """ Initialise class """
    
        self._initialise_objects()

        print('NingXiang Initialised')



    def _initialise_objects(self):
        
        self._seed = 18981124
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.clf_type = None
        self.ningxiang_output = None
        self.object_saving_address = None
        self.feature_importance = None
        

    
    def read_in_train_data(self, train_x, train_y, val_x = None, val_y = None):
        """ Reads in train data for building NingXiang output object """

        self.train_x = train_x
        print("Read in Train X data")

        self.train_y = train_y
        print("Read in Train y data")

        if val_x is not None:
            self.val_x = val_x
            print("Read in Val X data")
        
        if val_y is not None:
            self.val_y = val_y
            print("Read in Val y data")



    def set_model_type(self, type):
        """ Reads in underlying model object for tuning, and also read in what type of model it is """

        assert type == 'Classification' or type == 'Regression' # check

        self.clf_type = type 

        print(f'Successfully recorded model type: {self.clf_type}')



    def get_lr_based_feature_combinations(self, min_features = 0, gap = 1):
        """ Get NingXiang scores based on LR feature importance """

        if self.clf_type == None:
            raise AttributeError('clf_type not found, please run set_model_type() first')
            
        
        if self.clf_type == 'Classification':
            raise AttributeError(".get_lr_based_combinations() only works for continuous labels")
            

        if self.train_x is None or self.train_y is None:
            raise AttributeError('train_x and train_y not found, please run read_in_train_data')
            

        self.ningxiang_output = dict()

        curr_combo = list()
        remaining_features = list(self.train_x.columns)
        step = 0
        for i in range(len(remaining_features)):
            print(f'Up to {i+1}th variable')

            best_score = 0
            best_combo = None
            
            # try adding each new feature and getting lr
            for feature in remaining_features:
                tmp_combo = copy.deepcopy(curr_combo)
                tmp_combo.append(feature)

                lr = LinearRegression()
                lr.fit(self.train_x[tmp_combo], self.train_y)

                score = lr.score(self.train_x[tmp_combo], self.train_y)
                
                # if is new max of this round, then update
                if score > best_score:
                    added_feature = feature
                    best_score = score
                    best_combo = copy.deepcopy(tmp_combo)
            
            curr_combo = copy.deepcopy(best_combo)
            remaining_features.remove(added_feature) # remove added feature
            print(f'Current combination: {curr_combo}')
            print(f'Best score: {np.sqrt(best_score)}\n')
            
                    
            # store in ningxiang output
            step += 1

            if step % gap == 0 and i+1 >= min_features: # only add every gap-th combo, provided it has surpassed the min features

                step = 0
                
                self.ningxiang_output[tuple(curr_combo)] = np.sqrt(best_score)
            
        if len(list(self.ningxiang_output.keys())[-1]) != len(self.train_x.columns):
                    
            self.ningxiang_output[tuple(curr_combo)] = np.sqrt(best_score)  
        
        return self.ningxiang_output

    

    def get_rf_based_feature_combinations(self, min_features = 0, gap = 1, n_jobs = 1):
        """ Gets NingXiang scores based on RF feature importance """
        
        if self.clf_type == None:
            raise AttributeError('clf_type not found, please run set_model_type() first')
            

        if self.train_x is None or self.train_y is None:
            raise AttributeError('train_x and train_y not found, please run read_in_train_data')
            
        

        # Initialise the Random Forest objects
        if self.clf_type == 'Regression':
            self.rf = RandomForestRegressor(n_estimators = 100, max_depth = 12, max_features = 0.75, random_state = self._seed, ccp_alpha = 0, max_samples = 0.75, n_jobs = n_jobs)
        elif self.clf_type == 'Classification':
            self.rf = RandomForestClassifier(n_estimators = 100, max_depth = 12, max_features = 0.75, random_state = self._seed, ccp_alpha = 0, max_samples = 0.75, n_jobs = n_jobs)
        
        print('Begin fitting Random Forest')
        # fit the model and get the feature importances
        self.rf.fit(self.train_x, self.train_y)
        print('Finished fitting Random Forest')
        self.feature_importance = {self.train_x.columns[i]:self.rf.feature_importances_[i] for i in range(len(self.train_x.columns))}

        # use handle (which can be used on its own) to generate the ningxiang output
        self.ningxiang_output = self.get_rf_based_feature_combinations_from_feature_importance(self.feature_importance, min_features, gap)

        return self.ningxiang_output
    


    def get_rf_based_feature_combinations_from_feature_importance(self, feature_importance = None, min_features = 0, gap = 1):
        """ Takes in a dictionary of features:importance and returns feature importance"""
        
        feature_importance = None or self.feature_importance

        # sort features by feature importance (reversed order)
        feature_importance_sorted = self._sort_dict_reverse(feature_importance)
        # get the features in the output format that can be linked up with other JiaXing packages
        self.ningxiang_output = self._get_ningxiang_rf_output(feature_importance_sorted, min_features, gap)

        return self.ningxiang_output
        
    
    
    def _sort_dict_reverse(self, features):
        """ Helper to sort dictionary"""

        features_list = [(key, features[key]) for key in features]
        features_list.sort(key=lambda x:x[1], reverse=True)

        features_reverse_sorted = {x[0]:x[1] for x in features_list}

        return features_reverse_sorted

    


    def _get_ningxiang_rf_output(self, features_reverse_sorted, min_features, gap):
        """ Helper to get rf feature importance into ningxiang output format """

        out = dict() # NingXiang output object must be a dict

        feature_combo = list()
        score = 0

        first_time_past = True
        i = 0
        step = 0
        # Continuously add feature and its score
        for feature in features_reverse_sorted:
            feature_combo.append(feature)
            score += features_reverse_sorted[feature]

            combo = tuple(feature_combo)

            step += 1

            if step % gap == 0 and i+1 >= min_features: # only add every gap-th combo

                step = 0
                out[combo] = score

            i += 1
        
        if len(list(out.keys())[-1]) != len(features_reverse_sorted):
                    
            out[combo] = score

        return out



    def show_rf_stats(self):
        """ Shows feature importance dataframe """

        if self.feature_importance == None:
            raise AttributeError('Please run .show_rf_stats()')
        

        print(self.feature_importance)

        if self.val_x is not None and self.val_y is not None:
            print("Validation score:", self.rf.score(self.val_x, self.val_y))



    def _set_object_saving_address(self, address):
        """ Read in where to save the PuDong object """

        self.object_saving_address = address
        print('Successfully set object output address')

    

    def export_ningxiang_output(self, address):
        """ Export NingXiang's output object  """

        self._set_object_saving_address(address)

        # Export
        object_saving_address_split = self.object_saving_address.split('.pickle')[0]

        with open(f'{object_saving_address_split}.pickle', 'wb') as f:
            pickle.dump(self.ningxiang_output, f)