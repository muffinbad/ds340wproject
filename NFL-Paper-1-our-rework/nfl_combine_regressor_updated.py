# Updated version of `nfl_combine_regressor.py`
# This adds support for predicting Solo tackles, Sacks, and Touchdowns (TD) separately

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class nflCombineRegressor:

    def __init__(self):
        self.new_data = {}
        self.metrics = ['Solo', 'Sk', 'TD']  # Define new metrics to model

    def read_in_new(self, path):
        for year in ['2015', '2016', '2017']:
            file_path = os.path.join(path, f"{year}-new-data.xlsx")
            df = pd.read_excel(file_path, header=1)  # skip metadata row
            self.new_data[year] = df

    def prepare_data(self, metric):
        cols = ['40yd','Vertical','BP','Broad Jump','Shuttle','3Cone']
        x_all = []
        y_all = []

        for year, df in self.new_data.items():
            df = df.dropna(subset=cols + [metric])
            x = df[cols]
            y = df[metric].astype(float)

            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            x_all.append(pd.DataFrame(x_scaled, columns=cols))
            y_all.append(y)

        x_full = pd.concat(x_all)
        y_full = pd.concat(y_all)

        return train_test_split(x_full, y_full, train_size=0.8)

    def run_model_for_metric(self, metric):
        print(f"\nTraining model for {metric} prediction")
        x_train, x_test, y_train, y_test = self.prepare_data(metric)

        model = LinearRegression()
        model.fit(x_train, y_train)
        preds = model.predict(x_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print(f"RMSE for {metric}: {rmse:.4f}")
        print(f"R^2 Score for {metric}: {r2:.4f}")

        self.plot_feature_importance(model, x_train.columns, metric)

    def plot_feature_importance(self, model, columns, metric):
        importance = pd.Series(np.abs(model.coef_), index=columns).sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=importance, y=importance.index)
        plt.title(f'Feature Importance for {metric}', fontsize=16)
        plt.xlabel('Coefficient Magnitude')
        plt.tight_layout()
        plt.savefig(f'feature_imp_{metric}.png')
        plt.show()

if __name__ == '__main__':
    modeler = nflCombineRegressor()
    modeler.read_in_new("Data")  # Folder containing 2015-new-data.xlsx, etc.

    for metric in modeler.metrics:
        modeler.run_model_for_metric(metric)
