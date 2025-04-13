#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on [Date]

This script merges combine predictor data from the old NFL files:
  NFL 2013_edit.xlsx, NFL 2014_edit.xlsx, NFL 2015_edit.xlsx, NFL 2016_edit.xlsx, NFL 2017_edit.xlsx
with the target variable (TD) pulled from the new files:
  2015-new-data.xlsx, 2016-new-data.xlsx, 2017-new-data.xlsx.

Both datasets are merged on a common key. Regardless of the column names in the files,
we normalize them so that the key is "Player".
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

class nflCombineRegressor:
    def __init__(self):
        # Placeholders for combine data (old files)
        self.pd_2013 = None
        self.pd_2014 = None
        self.pd_2015 = None
        self.pd_2016 = None
        self.pd_2017 = None
        # Placeholders for new target data
        self.new_pd_2015 = None
        self.new_pd_2016 = None
        self.new_pd_2017 = None
        self.path = ""  # to store the base path

    def set_path(self, path):
        self.path = path

    def read_in(self, path):
        """
        Reads in the combine data from the old files.
        Afterwards, if the key column is named "Name" in any file, rename it to "Player".
        """
        file_2013 = os.path.join(path, "NFL 2013_edit.xlsx")
        file_2014 = os.path.join(path, "NFL 2014_edit.xlsx")
        file_2015 = os.path.join(path, "NFL 2015_edit.xlsx")
        file_2016 = os.path.join(path, "NFL 2016_edit.xlsx")
        file_2017 = os.path.join(path, "NFL 2017_edit.xlsx")

        self.pd_2013 = pd.read_excel(file_2013)
        self.pd_2014 = pd.read_excel(file_2014)
        self.pd_2015 = pd.read_excel(file_2015)
        self.pd_2016 = pd.read_excel(file_2016)
        self.pd_2017 = pd.read_excel(file_2017)
        
        # Normalize the unique identifier column in all old data files.
        for df in [self.pd_2013, self.pd_2014, self.pd_2015, self.pd_2016, self.pd_2017]:
            # If the key is "Name" but not "Player", then rename it to "Player"
            if "Name" in df.columns and "Player" not in df.columns:
                df.rename(columns={"Name": "Player"}, inplace=True)

    def load_new_data(self):
        """
        Reads in the new target files (with the TD column).
        Assumes that the header is on the second row (header=1) and cleans the column names.
        Also, if the key column appears as "player" in lowercase, we rename it to "Player".
        """
        file_2015_new = os.path.join(self.path, "2015-new-data.xlsx")
        file_2016_new = os.path.join(self.path, "2016-new-data.xlsx")
        file_2017_new = os.path.join(self.path, "2017-new-data.xlsx")
        
        self.new_pd_2015 = pd.read_excel(file_2015_new, header=1)
        self.new_pd_2016 = pd.read_excel(file_2016_new, header=1)
        self.new_pd_2017 = pd.read_excel(file_2017_new, header=1)
        
        # Clean column names (strip extra whitespace)
        for df in [self.new_pd_2015, self.new_pd_2016, self.new_pd_2017]:
            df.columns = df.columns.str.strip()
            # Normalize key: if 'player' (lowercase) exists, rename it to 'Player'
            if 'player' in df.columns and 'Player' not in df.columns:
                df.rename(columns={'player': 'Player'}, inplace=True)
        
        # Convert the TD column to numeric
        self.new_pd_2015['TD'] = pd.to_numeric(self.new_pd_2015['TD'], errors='coerce')
        self.new_pd_2016['TD'] = pd.to_numeric(self.new_pd_2016['TD'], errors='coerce')
        self.new_pd_2017['TD'] = pd.to_numeric(self.new_pd_2017['TD'], errors='coerce')
        
        print(len(self.new_pd_2015), "Target samples loaded for - 2015")
        print(len(self.new_pd_2016), "Target samples loaded for - 2016")
        print(len(self.new_pd_2017), "Target samples loaded for - 2017")

    def split_test(self):
        """
        Merges combine data (from old files) with target TD values (from new files) on the "Player" key,
        drops rows with missing predictor or target values or with TD equal to 0,
        scales the predictors, and splits the data into training (80%), validation (10%), and test (10%) sets.
        """
        common_key = "Player"  # now all files have the key "Player"

        cols = ['40yd','Vertical','BP','Broad Jump','Shuttle','3Cone']

        # --- Check for the common key in all DataFrames ---
        for df, name in [(self.pd_2015, "pd_2015"), (self.pd_2016, "pd_2016"), (self.pd_2017, "pd_2017"),
                         (self.new_pd_2015, "new_pd_2015"), (self.new_pd_2016, "new_pd_2016"), (self.new_pd_2017, "new_pd_2017")]:
            if common_key not in df.columns:
                raise KeyError(f"Common key '{common_key}' not found in {name}. Available columns: {list(df.columns)}")

        # Merge each year's combine data with the corresponding new target data on "Player"
        merged_2015 = pd.merge(self.pd_2015, self.new_pd_2015[[common_key, 'TD']], on=common_key, how='inner')
        merged_2016 = pd.merge(self.pd_2016, self.new_pd_2016[[common_key, 'TD']], on=common_key, how='inner')
        merged_2017 = pd.merge(self.pd_2017, self.new_pd_2017[[common_key, 'TD']], on=common_key, how='inner')

        # For each merged dataset, ensure TD is numeric, drop rows with missing values, and remove rows where TD == 0.
        for df in [merged_2015, merged_2016, merged_2017]:
            df['TD'] = pd.to_numeric(df['TD'], errors='coerce')
            df.dropna(subset=cols + ['TD'], inplace=True)
            df = df[df['TD'] != 0]

        X_2015 = merged_2015[cols]
        y_2015 = merged_2015['TD']
        X_2016 = merged_2016[cols]
        y_2016 = merged_2016['TD']
        X_2017 = merged_2017[cols]
        y_2017 = merged_2017['TD']

        print(len(X_2015), "Samples used for - 2015")
        print(len(X_2016), "Samples used for - 2016")
        print(len(X_2017), "Samples used for - 2017")

        X = pd.concat([X_2015, X_2016, X_2017])
        y = pd.concat([y_2015, y_2016, y_2017])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=cols, index=X.index)

        self.x_train, self.x_rem, self.y_train, self.y_rem = train_test_split(X, y, train_size=0.8)
        self.x_valid, self.x_test, self.y_valid, self.y_test = train_test_split(self.x_rem, self.y_rem, test_size=0.5)
     
    def model_test(self):
        """
        Evaluates several regressors using 10-fold cross-validation and then fits a final Linear Regression model on the test set.
        """
        GB = cross_validate(
            GradientBoostingRegressor(),
            self.x_train, self.y_train, cv=10,
            scoring=['neg_root_mean_squared_error'],
            return_train_score=True
        )
        RF = cross_validate(
            RandomForestRegressor(),
            self.x_train, self.y_train, cv=10,
            scoring=['neg_root_mean_squared_error'],
            return_train_score=True
        )
        LR = cross_validate(
            LinearRegression(),
            self.x_train, self.y_train, cv=10,
            scoring=['neg_root_mean_squared_error'],
            return_train_score=True
        )
        DT = cross_validate(
            DecisionTreeRegressor(),
            self.x_train, self.y_train, cv=10,
            scoring=['neg_root_mean_squared_error'],
            return_train_score=True
        )
        SV_R = cross_validate(
            SVR(),
            self.x_train, self.y_train, cv=10,
            scoring=['neg_root_mean_squared_error'],
            return_train_score=True
        )
        
        print('DT RMSE: ', np.abs(np.mean(DT['test_neg_root_mean_squared_error'])))
        print('GB RMSE: ', np.abs(np.mean(GB['test_neg_root_mean_squared_error'])))
        print('SVR RMSE: ', np.abs(np.mean(SV_R['test_neg_root_mean_squared_error'])))
        print('RF RMSE: ', np.abs(np.mean(RF['test_neg_root_mean_squared_error'])))
        print('LR RMSE: ', np.abs(np.mean(LR['test_neg_root_mean_squared_error'])))
        
        final_model = LinearRegression()
        final_model.fit(self.x_test, self.y_test)
        mse_value = mean_squared_error(self.y_test, final_model.predict(self.x_test))
        rmse_value = np.sqrt(mse_value)
        print('Final model RMSE on test data:', rmse_value)
        print('R squared on test data:', r2_score(self.y_test, final_model.predict(self.x_test)))
        return final_model
        
    def plot_feature_importance(self, final_model):
        """
        Uses the Linear Regression coefficients as feature importances and plots them.
        """
        importance = final_model.coef_
        print("Example coefficient:", importance[1])
        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i, v))

        feature_imp = pd.Series(np.abs(importance), index=self.x_test.columns).sort_values(ascending=False)
        fig, axs = plt.subplots(1, 1)
        sns.barplot(x=feature_imp, y=feature_imp.index)
        axs.set_title('Linear Regression Feature Importances', fontsize=20)
        axs.set_xlabel('Feature Importance (Beta Coefficient)', fontsize=16)
        axs.tick_params(axis='both', which='major', labelsize=16)
        axs.tick_params(axis='both', which='minor', labelsize=16)
        plt.draw()
        plt.show()
        fig.savefig("feature_imp_regression.png", dpi=150)

def main():
    parser = argparse.ArgumentParser(
        description="Run NFL Combine Regressor merging old combine data with new TD targets."
    )
    parser.add_argument(
        "--path", 
        type=str, 
        default="", 
        help="Folder path containing NFL_20xx_edit.xlsx files and new target files (2015-new-data.xlsx, etc.)"
    )
    args = parser.parse_args()

    start_time = time.time()
    regressor = nflCombineRegressor()
    regressor.set_path(args.path)
    regressor.read_in(args.path)
    regressor.load_new_data()
    regressor.split_test()
    final_model = regressor.model_test()
    regressor.plot_feature_importance(final_model)
    print("--- %s seconds ---" % (time.time()-start_time))

if __name__ == '__main__':
    main()
