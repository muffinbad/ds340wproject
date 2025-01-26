#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sept  28 11:16:50 2021
TO DO: remove data that does not have all combine features,
convert all snap data to binary
run classifiers:
1. RandomForestClassifier instead of Logistic Regression
2. Naive Bayes
3. K-Nearest Neighbors
4. Decision Tree
5. Support Vector Machines

@author: bszekely
"""

from nfl_combine_regressor import nflCombineRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


class nflCombineClassify(nflCombineRegressor):
    
    def __init__(self,path):
        super().__init__()
        # This calls the read_in from the parent, which now uses a relative path
        super().read_in(path)  #change this to be relative via argparse()
        super().cumulative_snaps()
        
    def snaps_to_binary(self):
        cols =['40yd','Vertical','BP','Broad Jump','Shuttle','3Cone']
        
        x_data_13_ = self.pd_2013[cols]
        x_data_14_ = self.pd_2014[cols]
        x_data_15_ = self.pd_2015[cols]
        x_data_16_ = self.pd_2015[cols]
        x_data_17_ = self.pd_2017[cols]

        index_nan_13 = x_data_13_.dropna().index.tolist()
        index_nan_14 = x_data_14_.dropna().index.tolist()
        index_nan_15 = x_data_15_.dropna().index.tolist()
        index_nan_16 = x_data_16_.dropna().index.tolist()
        index_nan_17 = x_data_17_.dropna().index.tolist()

        y_data_13_nonan = self.snaps_cum_2013.loc[index_nan_13]
        y_data_14_nonan = self.snaps_cum_2014.loc[index_nan_14]
        y_data_15_nonan = self.snaps_cum_2015.loc[index_nan_15]
        y_data_16_nonan = self.snaps_cum_2015.loc[index_nan_16]
        y_data_17_nonan = self.snaps_cum_2017.loc[index_nan_17]
        
        x_data_13_nonan = x_data_13_.loc[index_nan_13]
        x_data_14_nonan = x_data_14_.loc[index_nan_14]
        x_data_15_nonan = x_data_15_.loc[index_nan_15]
        x_data_16_nonan = x_data_16_.loc[index_nan_16]
        x_data_17_nonan = x_data_17_.loc[index_nan_17]
        
        print(len(y_data_13_nonan), "Samples ended with - 2013")
        print(len(y_data_14_nonan), "Samples ended with - 2014")
        print(len(y_data_15_nonan), "Samples ended with - 2015")
        print(len(y_data_16_nonan), "Samples ended with - 2016")
        print(len(y_data_17_nonan), "Samples ended with - 2017")
        
        #convert to binary
        y_data_13_nonan[y_data_13_nonan > 0] = 1
        y_data_14_nonan[y_data_14_nonan > 0] = 1
        y_data_15_nonan[y_data_15_nonan > 0] = 1
        y_data_16_nonan[y_data_16_nonan > 0] = 1
        y_data_17_nonan[y_data_17_nonan > 0] = 1
        
        len_snaps = (len(y_data_13_nonan) + len(y_data_14_nonan) + 
                     len(y_data_15_nonan) + len(y_data_16_nonan) + len(y_data_17_nonan))
        sum_1 = (sum(y_data_13_nonan) +  sum(y_data_14_nonan) + 
                 sum(y_data_15_nonan) + sum(y_data_16_nonan) + sum(y_data_17_nonan))
        
        print('ratio of how many 1 to 0: ', sum_1/len_snaps)

        scaler = StandardScaler()
        x_data_13 = scaler.fit_transform(x_data_13_nonan)
        x_data_14 = scaler.fit_transform(x_data_14_nonan) 
        x_data_15 = scaler.fit_transform(x_data_15_nonan)
        x_data_16 = scaler.fit_transform(x_data_16_nonan)
        x_data_17 = scaler.fit_transform(x_data_17_nonan)
        
        df_13 = pd.DataFrame(x_data_13, columns = cols)
        df_14 = pd.DataFrame(x_data_14, columns = cols)
        df_15 = pd.DataFrame(x_data_15, columns = cols)
        df_16 = pd.DataFrame(x_data_16, columns = cols)
        df_17 = pd.DataFrame(x_data_17, columns = cols)
        
        y = pd.concat([y_data_13_nonan, y_data_14_nonan, y_data_15_nonan, 
                       y_data_16_nonan, y_data_17_nonan]).astype(int)
        x = pd.concat([df_13, df_14, df_15, df_16, df_17])
        
        self.x_train_class, self.x_test_class, self.y_train_class, self.y_test_class = train_test_split(x,y, train_size=0.8) 

    def model_test_classify(self):
        DT = cross_validate(DecisionTreeClassifier(),
                            self.x_train_class, self.y_train_class, cv=10,
                            scoring=['accuracy'],return_train_score=True)
        GB = cross_validate(GradientBoostingClassifier(),
                            self.x_train_class, self.y_train_class, cv=10,
                            scoring=['accuracy'],return_train_score=True)
        SV_C = cross_validate(SVC(kernel='rbf'),
                              self.x_train_class, self.y_train_class, cv=10,
                              scoring=['accuracy'],return_train_score=True)
        RF = cross_validate(RandomForestClassifier(),
                            self.x_train_class, self.y_train_class, cv=10,
                            scoring=['accuracy'],return_train_score=True)
        LR = cross_validate(LogisticRegression(),
                            self.x_train_class, self.y_train_class, cv=10,
                            scoring=['accuracy'],return_train_score=True)

        print('results of DT: ',np.mean(DT['test_accuracy']))
        print('results of GB: ',np.mean(GB['test_accuracy']))
        print('results of SV_C: ',np.mean(SV_C['test_accuracy'])) 
        print('results of RF: ',np.mean(RF['test_accuracy'])) #WINNER
        print('results of LF: ',np.mean(LR['test_accuracy']))
        
        # Tune the winner (RandomForest) with RandomizedSearchCV
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(1, 110, num = 10)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        random_grid = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        }
            
        rf = RandomForestClassifier()
        test_model = RandomizedSearchCV(
            estimator = rf, 
            param_distributions = random_grid, 
            n_iter = 100, 
            cv = 10, 
            verbose=2, 
            n_jobs = -1
        )
        test_model.fit(self.x_train_class, self.y_train_class)
        print('print best params: ',test_model.best_params_)
        print('accuracy on test data: ',
              accuracy_score(self.y_test_class, test_model.predict(self.x_test_class)))
        return test_model

    def plot_feature_importance_classify(self, test_model):
        test_model_imp = pd.Series(
            test_model.best_estimator_.feature_importances_,
            index=self.x_test_class.columns
        ).sort_values(ascending=False)
        fig, axs = plt.subplots(1,1)
        sns.barplot(x=test_model_imp,y=test_model_imp.index)
        axs.set_xlabel('Feature Importance', fontsize=16)
        axs.set_title('Random Forest Classifier Feature Importance',fontsize=20)
        axs.tick_params(axis='both', which='major', labelsize=16)
        axs.tick_params(axis='both', which='minor', labelsize=16)
        plt.draw()
        plt.show()
            
if __name__ == '__main__':
    classify = nflCombineClassify('')
    classify.snaps_to_binary()
    test_model = classify.model_test_classify()
    classify.plot_feature_importance_classify(test_model)
