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

#updated all the new data that we read in
class nflCombineRegressor:
    def __init__(self):
        self.pd_2013 = None
        self.pd_2014 = None
        self.pd_2015 = None
        self.pd_2016 = None
        self.pd_2017 = None
        self.pd_2018 = None
        self.pd_2019 = None
        self.pd_2020 = None
        self.pd_2021 = None
        self.pd_2022 = None
        self.pd_2023 = None
        self.pd_2024 = None
        # Placeholders for new target data
        self.new_pd_2013 = None
        self.new_pd_2014 = None
        self.new_pd_2015 = None
        self.new_pd_2016 = None
        self.new_pd_2017 = None
        self.new_pd_2018 = None
        self.new_pd_2019 = None
        self.new_pd_2020 = None
        self.new_pd_2021 = None
        self.new_pd_2022 = None
        self.new_pd_2023 = None
        self.new_pd_2024 = None
        self.path = ""  # to store the base path

    def set_path(self, path):
        self.path = path

    def read_in(self, path):
        file_2013 = os.path.join(path, "NFL 2013_edit.xlsx")
        file_2014 = os.path.join(path, "NFL 2014_edit.xlsx")
        file_2015 = os.path.join(path, "NFL 2015_edit.xlsx")
        file_2016 = os.path.join(path, "NFL 2016_edit.xlsx")
        file_2017 = os.path.join(path, "NFL 2017_edit.xlsx")
        file_2018 = os.path.join(path, "NFL 2018_edit.xlsx")
        file_2019 = os.path.join(path, "NFL 2019_edit.xlsx")
        file_2020 = os.path.join(path, "NFL 2020_edit.xlsx")
        file_2021 = os.path.join(path, "NFL 2021_edit.xlsx")
        file_2022 = os.path.join(path, "NFL 2022_edit.xlsx")
        file_2023 = os.path.join(path, "NFL 2023_edit.xlsx")
        file_2024 = os.path.join(path, "NFL 2024_edit.xlsx")

        self.pd_2013 = pd.read_excel(file_2013)
        self.pd_2014 = pd.read_excel(file_2014)
        self.pd_2015 = pd.read_excel(file_2015)
        self.pd_2016 = pd.read_excel(file_2016)
        self.pd_2017 = pd.read_excel(file_2017)
        self.pd_2018 = pd.read_excel(file_2018)
        self.pd_2019 = pd.read_excel(file_2019)
        self.pd_2020 = pd.read_excel(file_2020)
        self.pd_2021 = pd.read_excel(file_2021)
        self.pd_2022 = pd.read_excel(file_2022)
        self.pd_2023 = pd.read_excel(file_2023)
        self.pd_2024 = pd.read_excel(file_2024)
        
        # Normalize the unique identifier column in all old data files.
        for df in [self.pd_2013, self.pd_2014, self.pd_2015, self.pd_2016, self.pd_2017, self.pd_2018, self.pd_2019, self.pd_2020, self.pd_2021, self.pd_2022, self.pd_2023, self.pd_2024]:
            if "Name" in df.columns and "Player" not in df.columns:
                df.rename(columns={"Name": "Player"}, inplace=True)

    def load_new_data(self):
        file_2013_new = os.path.join(self.path, "2013-new-data.xlsx")
        file_2014_new = os.path.join(self.path, "2014-new-data.xlsx")
        file_2015_new = os.path.join(self.path, "2015-new-data.xlsx")
        file_2016_new = os.path.join(self.path, "2016-new-data.xlsx")
        file_2017_new = os.path.join(self.path, "2017-new-data.xlsx")
        file_2018_new = os.path.join(self.path, "2018-new-data.xlsx")
        file_2019_new = os.path.join(self.path, "2019-new-data.xlsx")
        file_2020_new = os.path.join(self.path, "2020-new-data.xlsx")
        file_2021_new = os.path.join(self.path, "2021-new-data.xlsx")
        file_2022_new = os.path.join(self.path, "2022-new-data.xlsx")
        file_2023_new = os.path.join(self.path, "2023-new-data.xlsx")
        file_2024_new = os.path.join(self.path, "2024-new-data.xlsx")
        
        self.new_pd_2013 = pd.read_excel(file_2013_new, header=1)
        self.new_pd_2014 = pd.read_excel(file_2014_new, header=1)
        self.new_pd_2015 = pd.read_excel(file_2015_new, header=1)
        self.new_pd_2016 = pd.read_excel(file_2016_new, header=1)
        self.new_pd_2017 = pd.read_excel(file_2017_new, header=1)
        self.new_pd_2018 = pd.read_excel(file_2018_new, header=1)
        self.new_pd_2019 = pd.read_excel(file_2019_new, header=1)
        self.new_pd_2020 = pd.read_excel(file_2020_new, header=1)
        self.new_pd_2021 = pd.read_excel(file_2021_new, header=1)
        self.new_pd_2022 = pd.read_excel(file_2022_new, header=1)
        self.new_pd_2023 = pd.read_excel(file_2023_new, header=1)
        self.new_pd_2024 = pd.read_excel(file_2024_new, header=1)
        
        for df in [self.new_pd_2013, self.new_pd_2014, self.new_pd_2015, self.new_pd_2016, self.new_pd_2017, self.new_pd_2018, self.new_pd_2019, self.new_pd_2020, self.new_pd_2021, self.new_pd_2022, self.new_pd_2023, self.new_pd_2024]:
            df.columns = df.columns.str.strip()
            if 'player' in df.columns and 'Player' not in df.columns:
                df.rename(columns={'player': 'Player'}, inplace=True)
        
        for df in [self.new_pd_2013, self.new_pd_2014, self.new_pd_2015, self.new_pd_2016, self.new_pd_2017, self.new_pd_2018, self.new_pd_2019, self.new_pd_2020, self.new_pd_2021, self.new_pd_2022, self.new_pd_2023, self.new_pd_2024]:
            df['RUTD'] = pd.to_numeric(df['RUTD'], errors='coerce')
            df['RECTD'] = pd.to_numeric(df['RECTD'], errors='coerce')
            # Fill missing values with 0 and sum the two columns.
            df['TARGET'] = df['RUTD'].fillna(0) + df['RECTD'].fillna(0) #our measurement metric for touchdowns
        
        print(len(self.new_pd_2013), "Target samples loaded for - 2013")
        print(len(self.new_pd_2014), "Target samples loaded for - 2014")
        print(len(self.new_pd_2015), "Target samples loaded for - 2015")
        print(len(self.new_pd_2016), "Target samples loaded for - 2016")
        print(len(self.new_pd_2017), "Target samples loaded for - 2017")
        print(len(self.new_pd_2018), "Target samples loaded for - 2018")
        print(len(self.new_pd_2019), "Target samples loaded for - 2019")
        print(len(self.new_pd_2020), "Target samples loaded for - 2020")
        print(len(self.new_pd_2021), "Target samples loaded for - 2021")
        print(len(self.new_pd_2022), "Target samples loaded for - 2022")
        print(len(self.new_pd_2023), "Target samples loaded for - 2023")
        print(len(self.new_pd_2024), "Target samples loaded for - 2024")

    def split_test(self):
        common_key = "Player"
        cols = ['40yd','Vertical','BP','Broad Jump','Shuttle','3Cone']

        for df, name in [(self.pd_2013, "pd_2013"), (self.pd_2014, "pd_2014"), (self.pd_2015, "pd_2015"), (self.pd_2016, "pd_2016"), (self.pd_2017, "pd_2017"), (self.pd_2018, "pd_2018"), (self.pd_2019, "pd_2019"), (self.pd_2020, "pd_2020"), (self.pd_2021, "pd_2021"), (self.pd_2022, "pd_2022"), (self.pd_2023, "pd_2023"), (self.pd_2024, "pd_2024"),
                         (self.new_pd_2013, "new_pd_2013"), (self.new_pd_2014, "new_pd_2014"), (self.new_pd_2015, "new_pd_2015"), (self.new_pd_2016, "new_pd_2016"), (self.new_pd_2017, "new_pd_2017"), (self.new_pd_2018, "new_pd_2018"), (self.new_pd_2019, "new_pd_2019"), (self.new_pd_2020, "new_pd_2020"), (self.new_pd_2021, "new_pd_2021"), (self.new_pd_2022, "new_pd_2022"), (self.new_pd_2023, "new_pd_2023"), (self.new_pd_2024, "new_pd_2024")]:
            if common_key not in df.columns:
                raise KeyError(f"Common key '{common_key}' not found in {name}. Available columns: {list(df.columns)}")
        
        merged_2013 = pd.merge(self.pd_2013, self.new_pd_2013[[common_key, 'TARGET']], on=common_key, how='inner')
        merged_2014 = pd.merge(self.pd_2014, self.new_pd_2014[[common_key, 'TARGET']], on=common_key, how='inner')
        merged_2015 = pd.merge(self.pd_2015, self.new_pd_2015[[common_key, 'TARGET']], on=common_key, how='inner')
        merged_2016 = pd.merge(self.pd_2016, self.new_pd_2016[[common_key, 'TARGET']], on=common_key, how='inner')
        merged_2017 = pd.merge(self.pd_2017, self.new_pd_2017[[common_key, 'TARGET']], on=common_key, how='inner')
        merged_2018 = pd.merge(self.pd_2018, self.new_pd_2018[[common_key, 'TARGET']], on=common_key, how='inner')
        merged_2019 = pd.merge(self.pd_2019, self.new_pd_2019[[common_key, 'TARGET']], on=common_key, how='inner')
        merged_2020 = pd.merge(self.pd_2020, self.new_pd_2020[[common_key, 'TARGET']], on=common_key, how='inner')
        merged_2021 = pd.merge(self.pd_2021, self.new_pd_2021[[common_key, 'TARGET']], on=common_key, how='inner')
        merged_2022 = pd.merge(self.pd_2022, self.new_pd_2022[[common_key, 'TARGET']], on=common_key, how='inner')
        merged_2023 = pd.merge(self.pd_2023, self.new_pd_2023[[common_key, 'TARGET']], on=common_key, how='inner')
        merged_2024 = pd.merge(self.pd_2024, self.new_pd_2024[[common_key, 'TARGET']], on=common_key, how='inner')

        for df in [merged_2013, merged_2014, merged_2015, merged_2016, merged_2017, merged_2018, merged_2019, merged_2020, merged_2021, merged_2022, merged_2023, merged_2024]:
            df['TARGET'] = pd.to_numeric(df['TARGET'], errors='coerce')
            df.dropna(subset=cols + ['TARGET'], inplace=True)
            df = df[df['TARGET'] != 0]

        X_2013 = merged_2013[cols]
        y_2013 = merged_2013['TARGET']
        X_2014 = merged_2014[cols]
        y_2014 = merged_2014['TARGET']
        X_2015 = merged_2015[cols]
        y_2015 = merged_2015['TARGET']
        X_2016 = merged_2016[cols]
        y_2016 = merged_2016['TARGET']
        X_2017 = merged_2017[cols]
        y_2017 = merged_2017['TARGET']
        X_2018 = merged_2018[cols]
        y_2018 = merged_2018['TARGET']
        X_2019 = merged_2019[cols]
        y_2019 = merged_2019['TARGET']
        X_2020 = merged_2020[cols]
        y_2020 = merged_2020['TARGET']
        X_2021 = merged_2021[cols]
        y_2021 = merged_2021['TARGET']
        X_2022 = merged_2022[cols]
        y_2022 = merged_2022['TARGET']
        X_2023 = merged_2023[cols]
        y_2023 = merged_2023['TARGET']
        X_2024 = merged_2024[cols]
        y_2024 = merged_2024['TARGET']

        print(len(X_2013), "Samples for - 2013")
        print(len(X_2014), "Samples for - 2014")
        print(len(X_2015), "Samples for - 2015")
        print(len(X_2016), "Samples for - 2016")
        print(len(X_2017), "Samples for - 2017")
        print(len(X_2018), "Samples for - 2018")
        print(len(X_2019), "Samples for - 2019")
        print(len(X_2020), "Samples for - 2020")
        print(len(X_2021), "Samples for - 2021")
        print(len(X_2022), "Samples for - 2022")
        print(len(X_2023), "Samples for - 2023")
        print(len(X_2024), "Samples for - 2024")

        X = pd.concat([X_2013, X_2014, X_2015, X_2016, X_2017, X_2018, X_2019, X_2020, X_2021, X_2022, X_2023, X_2024])
        y = pd.concat([y_2013, y_2014, y_2015, y_2016, y_2017, y_2018, y_2019, y_2020, y_2021, y_2022, y_2023, y_2024])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=cols, index=X.index)

        self.x_train, self.x_rem, self.y_train, self.y_rem = train_test_split(X, y, train_size=0.8)
        self.x_valid, self.x_test, self.y_valid, self.y_test = train_test_split(self.x_rem, self.y_rem, test_size=0.5)
     
    def model_test(self):
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
        importance = final_model.coef_

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
        description="Run NFL Combine Regressor merging old combine data with new TARGET (RUTD + RECTD) values."
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
