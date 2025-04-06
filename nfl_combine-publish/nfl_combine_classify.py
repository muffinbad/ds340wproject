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

import argparse
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

# Import parent class
from nfl_combine_regressor import nflCombineRegressor

class nflCombineClassify(nflCombineRegressor):
    
    def __init__(self, path):
        super().__init__()
        # Read the data (adjusted to use path)
        super().read_in(path)

        # Convert numeric columns that might contain commas to floats:
        for df in [self.pd_2013, self.pd_2014, self.pd_2015, self.pd_2016, self.pd_2017]:
            for col in ['40yd','Vertical','BP','Broad Jump','Shuttle','3Cone']:
                if col in df.columns:
                    # Replace commas (if any) with periods
                    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        # Compute cumulative snaps
        super().cumulative_snaps()
        
    def snaps_to_binary(self):
        cols = ['40yd','Vertical','BP','Broad Jump','Shuttle','3Cone']
        
        # Pull the relevant columns for each year
        x_data_13_ = self.pd_2013[cols]
        x_data_14_ = self.pd_2014[cols]
        x_data_15_ = self.pd_2015[cols]
        x_data_16_ = self.pd_2016[cols]  # fixed to use 2016
        x_data_17_ = self.pd_2017[cols]

        # Drop rows with NaN for these columns
        idx_13 = x_data_13_.dropna().index
        idx_14 = x_data_14_.dropna().index
        idx_15 = x_data_15_.dropna().index
        idx_16 = x_data_16_.dropna().index
        idx_17 = x_data_17_.dropna().index

        # Grab the snaps for those same rows
        y_13 = self.snaps_cum_2013.loc[idx_13]
        y_14 = self.snaps_cum_2014.loc[idx_14]
        y_15 = self.snaps_cum_2015.loc[idx_15]
        y_16 = self.snaps_cum_2016.loc[idx_16]
        y_17 = self.snaps_cum_2017.loc[idx_17]
        
        # Also subset the combine columns
        x_13_nonan = x_data_13_.loc[idx_13]
        x_14_nonan = x_data_14_.loc[idx_14]
        x_15_nonan = x_data_15_.loc[idx_15]
        x_16_nonan = x_data_16_.loc[idx_16]
        x_17_nonan = x_data_17_.loc[idx_17]
        
        # Print how many samples remain
        print(len(y_13), "Samples ended with - 2013")
        print(len(y_14), "Samples ended with - 2014")
        print(len(y_15), "Samples ended with - 2015")
        print(len(y_16), "Samples ended with - 2016")
        print(len(y_17), "Samples ended with - 2017")
        
        # Convert snaps to binary: 1 if >0 else 0
        y_13[y_13 > 0] = 1
        y_14[y_14 > 0] = 1
        y_15[y_15 > 0] = 1
        y_16[y_16 > 0] = 1
        y_17[y_17 > 0] = 1
        
        # Merge all years
        len_snaps = len(y_13) + len(y_14) + len(y_15) + len(y_16) + len(y_17)
        sum_1 = y_13.sum() + y_14.sum() + y_15.sum() + y_16.sum() + y_17.sum()
        print('ratio of how many 1 to 0: ', sum_1 / len_snaps)

        # Scale each year's data separately, as in the original
        scaler = StandardScaler()
        x_13_scaled = scaler.fit_transform(x_13_nonan)
        x_14_scaled = scaler.fit_transform(x_14_nonan)
        x_15_scaled = scaler.fit_transform(x_15_nonan)
        x_16_scaled = scaler.fit_transform(x_16_nonan)
        x_17_scaled = scaler.fit_transform(x_17_nonan)
        
        df_13 = pd.DataFrame(x_13_scaled, columns=cols, index=idx_13)
        df_14 = pd.DataFrame(x_14_scaled, columns=cols, index=idx_14)
        df_15 = pd.DataFrame(x_15_scaled, columns=cols, index=idx_15)
        df_16 = pd.DataFrame(x_16_scaled, columns=cols, index=idx_16)
        df_17 = pd.DataFrame(x_17_scaled, columns=cols, index=idx_17)
        
        # Concatenate all years
        X = pd.concat([df_13, df_14, df_15, df_16, df_17])
        Y = pd.concat([y_13, y_14, y_15, y_16, y_17]).astype(int)
        
        # Split data
        self.x_train_class, self.x_test_class, self.y_train_class, self.y_test_class = \
            train_test_split(X, Y, train_size=0.8) 

    def model_test_classify(self):
        """
        Cross-validate several classifiers and print mean accuracy scores,
        then tune a RandomForest via RandomizedSearchCV, and return the best model.
        """
        DT = cross_validate(
            DecisionTreeClassifier(), 
            self.x_train_class, 
            self.y_train_class,
            cv=10, 
            scoring=['accuracy'], 
            return_train_score=True
        )
        GB = cross_validate(
            GradientBoostingClassifier(),
            self.x_train_class, 
            self.y_train_class,
            cv=10, 
            scoring=['accuracy'], 
            return_train_score=True
        )
        SV_C = cross_validate(
            SVC(kernel='rbf'),
            self.x_train_class, 
            self.y_train_class,
            cv=10, 
            scoring=['accuracy'], 
            return_train_score=True
        )
        RF = cross_validate(
            RandomForestClassifier(),
            self.x_train_class, 
            self.y_train_class,
            cv=10, 
            scoring=['accuracy'], 
            return_train_score=True
        )
        LR = cross_validate(
            LogisticRegression(),
            self.x_train_class, 
            self.y_train_class,
            cv=10, 
            scoring=['accuracy'], 
            return_train_score=True
        )

        # Print cross_val mean accuracies (matching original script output)
        print('results of DT: ', np.mean(DT['test_accuracy']))
        print('results of GB: ', np.mean(GB['test_accuracy']))
        print('results of SV_C: ', np.mean(SV_C['test_accuracy']))
        print('results of RF: ', np.mean(RF['test_accuracy']), '#WINNER')
        print('results of LF: ', np.mean(LR['test_accuracy']))

        # Hyperparameter space for RandomForest (same as original)
        n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]
        max_features = ['sqrt', 'log2', None]
        max_depth = [int(x) for x in np.linspace(1, 110, num=10)]
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
        
        # RandomizedSearchCV for the best RandomForest
        rf = RandomForestClassifier()
        test_model = RandomizedSearchCV(
            estimator=rf,
            param_distributions=random_grid,
            n_iter=100,
            cv=10,
            verbose=2,
            n_jobs=-1
        )
        test_model.fit(self.x_train_class, self.y_train_class)
        print('print best params: ', test_model.best_params_)

        test_pred = test_model.predict(self.x_test_class)
        print('accuracy on test data: ', accuracy_score(self.y_test_class, test_pred))

        return test_model

    def plot_feature_importance_classify(self, test_model):
        test_model_imp = pd.Series(
            test_model.best_estimator_.feature_importances_,
            index=self.x_test_class.columns
        ).sort_values(ascending=False)
        fig, ax = plt.subplots(1, 1)
        sns.barplot(x=test_model_imp, y=test_model_imp.index)
        ax.set_xlabel('Feature Importance', fontsize=16)
        ax.set_title('Random Forest Classifier Feature Importance', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)
        plt.draw()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run NFL Combine Classifier")
    parser.add_argument(
        "--path", 
        type=str, 
        default="", 
        help="Folder path with NFL 20xx_edit.xlsx files (defaults to current)."
    )
    args = parser.parse_args()

    classify = nflCombineClassify(args.path)
    classify.snaps_to_binary()
    test_model = classify.model_test_classify()
    classify.plot_feature_importance_classify(test_model)

if __name__ == '__main__':
    main()
