# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 22:56:18 2021

TODO: how many models to we want? import all data across all seasons

TODO: make a separate script that assesses whether the combine metrics
predicts on whether a player will play within the next 4 seasons - a classification
question

I am currently working on aggregating the 2013 and 2014 data together

add what you think we need. This is just the base framework. 

@author: Brian
"""

import os
import pandas as pd
import numpy as np
pd.options.plotting.backend = 'holoviews'
import hvplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time
from sklearn.model_selection import cross_validate

class nflCombineRegressor:

    def __init__(self):
        snaps_cum_2013 = pd.Series(dtype = float)
        snaps_cum_2014 = pd.Series(dtype = float)
        snaps_cum_2015 = pd.Series(dtype = float)
        snaps_cum_2016 = pd.Series(dtype = float)
        snaps_cum_2017 = pd.Series(dtype = float)

    def read_in(self,path): #change this to be relative via argparse() or local folder
        # Build the filepaths using os.path.join (if path="" then it uses current folder)
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
        
        self.snaps_2013 = pd.read_excel(file_2013, sheet_name="Snaps")
        self.snaps_2014 = pd.read_excel(file_2014, sheet_name="Snaps")
        self.snaps_2015 = pd.read_excel(file_2015, sheet_name="Snaps")
        self.snaps_2016 = pd.read_excel(file_2016, sheet_name="Snaps")
        self.snaps_2017 = pd.read_excel(file_2017, sheet_name="Snaps")

    def cumulative_snaps(self):
        """
        Sums all data across seasons, defense, offense, and special teams.
        Only numeric columns are summed, to avoid string concatenation issues.
        """
        self.snaps_cum_2013 = self.snaps_2013.select_dtypes(include=np.number).sum(axis=1)
        self.snaps_cum_2014 = self.snaps_2014.select_dtypes(include=np.number).sum(axis=1)
        self.snaps_cum_2015 = self.snaps_2015.select_dtypes(include=np.number).sum(axis=1)
        self.snaps_cum_2016 = self.snaps_2016.select_dtypes(include=np.number).sum(axis=1)
        self.snaps_cum_2017 = self.snaps_2017.select_dtypes(include=np.number).sum(axis=1)
        
        print(len(self.snaps_cum_2013), "Samples started with - 2013")
        print(len(self.snaps_cum_2014), "Samples started with - 2014")
        print(len(self.snaps_cum_2015), "Samples started with - 2015")
        print(len(self.snaps_cum_2016), "Samples started with - 2016")
        print(len(self.snaps_cum_2017), "Samples started with - 2017")
    
    def split_test(self):
        index_nonzero_13 = self.snaps_cum_2013[self.snaps_cum_2013 !=0 ].index.tolist()
        index_nonzero_14 = self.snaps_cum_2014[self.snaps_cum_2014 !=0 ].index.tolist()
        index_nonzero_15 = self.snaps_cum_2015[self.snaps_cum_2015 !=0 ].index.tolist()
        index_nonzero_16 = self.snaps_cum_2016[self.snaps_cum_2016 !=0 ].index.tolist()
        index_nonzero_17 = self.snaps_cum_2017[self.snaps_cum_2017 !=0 ].index.tolist()

        snaps_parse_13 = self.snaps_cum_2013.iloc[index_nonzero_13]
        snaps_parse_14 = self.snaps_cum_2014.iloc[index_nonzero_14]
        snaps_parse_15 = self.snaps_cum_2015.iloc[index_nonzero_15]
        snaps_parse_16 = self.snaps_cum_2016.iloc[index_nonzero_16]
        snaps_parse_17 = self.snaps_cum_2017.iloc[index_nonzero_17]
        
        pd_2013_nozero = self.pd_2013.iloc[index_nonzero_13,:]
        pd_2014_nozero = self.pd_2014.iloc[index_nonzero_14,:]
        pd_2015_nozero = self.pd_2015.iloc[index_nonzero_15,:]
        pd_2016_nozero = self.pd_2016.iloc[index_nonzero_16,:]
        pd_2017_nozero = self.pd_2017.iloc[index_nonzero_17,:]
        
        cols = ['40yd','Vertical','BP','Broad Jump','Shuttle','3Cone']
        
        x_data_13_ = pd_2013_nozero[cols]
        x_data_14_ = pd_2014_nozero[cols]
        x_data_15_ = pd_2015_nozero[cols]
        x_data_16_ = pd_2016_nozero[cols]
        x_data_17_ = pd_2017_nozero[cols]

        index_nan_13 = x_data_13_.dropna().index.tolist()
        index_nan_14 = x_data_14_.dropna().index.tolist()
        index_nan_15 = x_data_15_.dropna().index.tolist()
        index_nan_16 = x_data_16_.dropna().index.tolist()
        index_nan_17 = x_data_17_.dropna().index.tolist()
    
        x_data_13_nonan = x_data_13_.loc[index_nan_13]
        x_data_14_nonan = x_data_14_.loc[index_nan_14]
        x_data_15_nonan = x_data_15_.loc[index_nan_15]
        x_data_16_nonan = x_data_16_.loc[index_nan_16]
        x_data_17_nonan = x_data_17_.loc[index_nan_17]
        
        y_data_13_nonan = snaps_parse_13.loc[index_nan_13]
        y_data_14_nonan = snaps_parse_14.loc[index_nan_14]
        y_data_15_nonan = snaps_parse_15.loc[index_nan_15]
        y_data_16_nonan = snaps_parse_16.loc[index_nan_16]
        y_data_17_nonan = snaps_parse_17.loc[index_nan_17]

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
        
        x = pd.concat([df_13, df_14, df_15, df_16, df_17]) 
        y = pd.concat([y_data_13_nonan, y_data_14_nonan, y_data_15_nonan, y_data_16_nonan, y_data_17_nonan])     

        print(len(x_data_13_nonan), "Samples started with - 2013")
        print(len(x_data_14_nonan), "Samples started with - 2014")
        print(len(x_data_15_nonan), "Samples started with - 2015")
        print(len(x_data_16_nonan), "Samples started with - 2016")
        print(len(x_data_17_nonan), "Samples started with - 2017")

        self.x_train, self.x_rem, self.y_train, self.y_rem = train_test_split(x,y, train_size=0.8)
        self.x_valid, self.x_test, self.y_valid, self.y_test = train_test_split(self.x_rem,self.y_rem, test_size=0.5)
     
    def model_test(self):
        GB = cross_validate(GradientBoostingRegressor(),
                            self.x_train, self.y_train, cv=10,
                            scoring=['neg_root_mean_squared_error'],
                            return_train_score=True)
        RF = cross_validate(RandomForestRegressor(),
                            self.x_train, self.y_train, cv=10,
                            scoring=['neg_root_mean_squared_error'],
                            return_train_score=True)
        LR = cross_validate(LinearRegression(),
                            self.x_train, self.y_train, cv=10,
                            scoring=['neg_root_mean_squared_error'],
                            return_train_score=True)
        DT = cross_validate(DecisionTreeRegressor(),
                            self.x_train, self.y_train, cv=10,
                            scoring=['neg_root_mean_squared_error'],
                            return_train_score=True)
        SV_R = cross_validate(SVR(),
                              self.x_train, self.y_train, cv=10,
                              scoring=['neg_root_mean_squared_error'],
                              return_train_score=True)
        
        print('results of DT: ',np.abs(np.mean(DT['test_neg_root_mean_squared_error'])))
        print('results of GB: ',np.abs(np.mean(GB['test_neg_root_mean_squared_error'])))
        print('results of SV_R: ',np.abs(np.mean(SV_R['test_neg_root_mean_squared_error'])))
        print('results of RF: ',np.abs(np.mean(RF['test_neg_root_mean_squared_error'])))
        print('results of LR: ',np.abs(np.mean(LR['test_neg_root_mean_squared_error']))) #winner
        
        final_model = LinearRegression()
        final_model.fit(self.x_test,self.y_test)
        # Instead of squared=False, do a manual sqrt
        mse_value = mean_squared_error(self.y_test, final_model.predict(self.x_test))
        rmse_value = np.sqrt(mse_value)
        print('RMSE on test data:', rmse_value)
        print('R squared value test data:', r2_score(self.y_test, final_model.predict(self.x_test)))
        return final_model
        
    def plot_feature_importance(self, final_model):
        importance = final_model.coef_
        print(importance[1])
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        #Calculate feature importance by absolute value of the coefficients
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

if __name__ == '__main__':
    start_time = time.time()
    nfl = nflCombineRegressor()
    # If the Excel files are in the same folder, use path="" or "."
    nfl.read_in("")
    nfl.cumulative_snaps()
    nfl.split_test()
    final_model = nfl.model_test()
    nfl.plot_feature_importance(final_model)
    print("--- %s seconds ---" % (time.time() - start_time))
