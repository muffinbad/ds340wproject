import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE #SMOTE
from nfl_combine_regressor import nflCombineRegressor

class nflCombineClassify(nflCombineRegressor):
    
    def __init__(self, path):
        super().__init__()
        self.set_path(path)
        super().read_in(path)
        super().load_new_data()
        
        cols = ['40yd','Vertical','BP','Broad Jump','Shuttle','3Cone']
        for df in [self.pd_2013, self.pd_2014, self.pd_2015, self.pd_2016, self.pd_2017]:
            if "Name" in df.columns and "Player" not in df.columns:
                df.rename(columns={"Name": "Player"}, inplace=True)
            for col in cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
    def snaps_to_binary(self):
        cols = ['40yd','Vertical','BP','Broad Jump','Shuttle','3Cone']
        common_key = "Player"

        for df, name in [(self.pd_2013, "pd_2013"), (self.pd_2014, "pd_2014"), (self.pd_2015, "pd_2015"), (self.pd_2016, "pd_2016"), (self.pd_2017, "pd_2017"),
                         (self.new_pd_2013, "new_pd_2013"), (self.new_pd_2014, "new_pd_2014"), (self.new_pd_2015, "new_pd_2015"), (self.new_pd_2016, "new_pd_2016"), (self.new_pd_2017, "new_pd_2017")]:
            if common_key not in df.columns:
                raise KeyError(f"Common key '{common_key}' not found in {name}. Available columns: {list(df.columns)}")

        merged_2013 = pd.merge(self.pd_2013, self.new_pd_2013[[common_key, 'TARGET']], on=common_key, how='inner')
        merged_2014 = pd.merge(self.pd_2014, self.new_pd_2014[[common_key, 'TARGET']], on=common_key, how='inner')
        merged_2015 = pd.merge(self.pd_2015, self.new_pd_2015[[common_key, 'TARGET']], on=common_key, how='inner')
        merged_2016 = pd.merge(self.pd_2016, self.new_pd_2016[[common_key, 'TARGET']], on=common_key, how='inner')
        merged_2017 = pd.merge(self.pd_2017, self.new_pd_2017[[common_key, 'TARGET']], on=common_key, how='inner')
        
        for df in [merged_2013, merged_2014, merged_2015, merged_2016, merged_2017]:
            df['TARGET'] = pd.to_numeric(df['TARGET'], errors='coerce')
            df.dropna(subset=cols + ['TARGET'], inplace=True)
        
        merged_2013['TARGET'] = (merged_2013['TARGET'] > 0).astype(int)
        merged_2014['TARGET'] = (merged_2014['TARGET'] > 0).astype(int)
        merged_2015['TARGET'] = (merged_2015['TARGET'] > 0).astype(int)
        merged_2016['TARGET'] = (merged_2016['TARGET'] > 0).astype(int)
        merged_2017['TARGET'] = (merged_2017['TARGET'] > 0).astype(int)
        
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
        
        print(len(X_2013), "Samples for - 2013")
        print(len(X_2014), "Samples for - 2014")
        print(len(X_2015), "Samples for - 2015")
        print(len(X_2016), "Samples for - 2016")
        print(len(X_2017), "Samples for - 2017")
        
        X = pd.concat([X_2013, X_2014, X_2015, X_2016, X_2017])
        y = pd.concat([y_2013, y_2014, y_2015, y_2016, y_2017])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=cols, index=X.index)
        
        # Split into training and test sets (using stratification).
        self.x_train_class, self.x_test_class, self.y_train_class, self.y_test_class = train_test_split(
            X, y, train_size=0.8, stratify=y, random_state=42
        )
        
        # Apply SMOTE on the training set to balance classes.
        sm = SMOTE(random_state=42)
        self.x_train_class, self.y_train_class = sm.fit_resample(self.x_train_class, self.y_train_class)
    
    def model_test_classify(self):
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

        print('DT accuracy: ', np.mean(DT['test_accuracy']))
        print('GB accuracy: ', np.mean(GB['test_accuracy']))
        print('SVC accuracy: ', np.mean(SV_C['test_accuracy']))
        print('RF accuracy: ', np.mean(RF['test_accuracy']), '#WINNER')
        print('LR accuracy: ', np.mean(LR['test_accuracy']))

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
        print('Best parameters:', test_model.best_params_)

        test_pred = test_model.predict(self.x_test_class)
        print('Test accuracy:', accuracy_score(self.y_test_class, test_pred))

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
    parser = argparse.ArgumentParser(
        description="Run NFL Combine Classifier with merged combine data (old files) and binary TARGET labels (RUTD + RECTD) from new files."
    )
    parser.add_argument(
        "--path", 
        type=str, 
        default="", 
        help="Folder path containing NFL_20xx_edit.xlsx files and new target files (2015-new-data.xlsx, etc.)"
    )
    args = parser.parse_args()

    classifier = nflCombineClassify(args.path)
    classifier.snaps_to_binary()
    test_model = classifier.model_test_classify()
    classifier.plot_feature_importance_classify(test_model)

if __name__ == '__main__':
    main()
