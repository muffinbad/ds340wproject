# Updated version of `nfl_combine_classify.py`
# This runs binary classification models for Solo, Sk, and TD individually

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class nflCombineClassify:

    def __init__(self):
        self.new_data = {}
        self.metrics = ['Solo', 'Sk', 'TD']

    def read_in_new(self, path):
        for year in ['2015', '2016', '2017']:
            file_path = os.path.join(path, f"{year}-new-data.xlsx")
            df = pd.read_excel(file_path, header=1)  # skip header row
            self.new_data[year] = df

    def prepare_binary_data(self, metric):
        cols = ['40yd','Vertical','BP','Broad Jump','Shuttle','3Cone']
        x_all = []
        y_all = []

        for year, df in self.new_data.items():
            df = df.dropna(subset=cols + [metric])
            x = df[cols]
            y = (df[metric] > 0).astype(int)  # Binary classification: 1 if value > 0

            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)

            x_all.append(pd.DataFrame(x_scaled, columns=cols))
            y_all.append(y)

        x_full = pd.concat(x_all)
        y_full = pd.concat(y_all)

        return train_test_split(x_full, y_full, train_size=0.8)

    def run_classifier_for_metric(self, metric):
        print(f"\nRunning classifier for binary {metric}")
        x_train, x_test, y_train, y_test = self.prepare_binary_data(metric)

        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        preds = rf.predict(x_test)

        acc = accuracy_score(y_test, preds)
        print(f"Accuracy on test data for {metric}: {acc:.4f}")

        self.plot_feature_importance(rf, x_train.columns, metric)

    def plot_feature_importance(self, model, columns, metric):
        importance = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=importance, y=importance.index)
        plt.title(f'Feature Importance for {metric} (Binary)', fontsize=16)
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'classifier_imp_{metric}.png')
        plt.show()

if __name__ == '__main__':
    clf = nflCombineClassify()
    clf.read_in_new("Data")  # Folder containing new Excel files

    for metric in clf.metrics:
        clf.run_classifier_for_metric(metric)
