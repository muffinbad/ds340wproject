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
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

from nfl_combine_regressor_Tackles import nflCombineRegressor 


class nflCombineClassify(nflCombineRegressor):
    def __init__(self, path):
        super().__init__()
        self.set_path(path)
        super().read_in(path)
        super().load_new_data()

        # ensure numeric combine metrics
        cols = ["40yd", "Vertical", "BP", "Broad Jump", "Shuttle", "3Cone"]
        for df in [
            self.pd_2013, self.pd_2014, self.pd_2015, self.pd_2016, self.pd_2017, self.pd_2018,
            self.pd_2019, self.pd_2020, self.pd_2021, self.pd_2022, self.pd_2023, self.pd_2024
        ]:
            if "Name" in df.columns and "Player" not in df.columns:
                df.rename(columns={"Name": "Player"}, inplace=True)
            for col in cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
                    df[col] = pd.to_numeric(df[col], errors="coerce")


    def snaps_to_binary(self):
        cols = ["40yd", "Vertical", "BP", "Broad Jump", "Shuttle", "3Cone"]
        key = "Player"

        m13 = pd.merge(self.pd_2013, self.new_pd_2013[[key, "TARGET"]], on=key, how="inner")
        m14 = pd.merge(self.pd_2014, self.new_pd_2014[[key, "TARGET"]], on=key, how="inner")
        m15 = pd.merge(self.pd_2015, self.new_pd_2015[[key, "TARGET"]], on=key, how="inner")
        m16 = pd.merge(self.pd_2016, self.new_pd_2016[[key, "TARGET"]], on=key, how="inner")
        m17 = pd.merge(self.pd_2017, self.new_pd_2017[[key, "TARGET"]], on=key, how="inner")
        m18 = pd.merge(self.pd_2018, self.new_pd_2018[[key, "TARGET"]], on=key, how="inner")
        m19 = pd.merge(self.pd_2019, self.new_pd_2019[[key, "TARGET"]], on=key, how="inner")
        m20 = pd.merge(self.pd_2020, self.new_pd_2020[[key, "TARGET"]], on=key, how="inner")
        m21 = pd.merge(self.pd_2021, self.new_pd_2021[[key, "TARGET"]], on=key, how="inner")
        m22 = pd.merge(self.pd_2022, self.new_pd_2022[[key, "TARGET"]], on=key, how="inner")
        m23 = pd.merge(self.pd_2023, self.new_pd_2023[[key, "TARGET"]], on=key, how="inner")
        m24 = pd.merge(self.pd_2024, self.new_pd_2024[[key, "TARGET"]], on=key, how="inner")

        for df in [m13, m14, m15, m16, m17, m18, m19, m20, m21, m22, m23, m24]:
            df["TARGET"] = pd.to_numeric(df["TARGET"], errors="coerce")
            df.dropna(subset=cols + ["TARGET"], inplace=True)
            df["TARGET"] = (df["TARGET"] > 0).astype(int)

        X_13, y_13 = m13[cols], m13["TARGET"]
        X_14, y_14 = m14[cols], m14["TARGET"]
        X_15, y_15 = m15[cols], m15["TARGET"]
        X_16, y_16 = m16[cols], m16["TARGET"]
        X_17, y_17 = m17[cols], m17["TARGET"]
        X_18, y_18 = m18[cols], m18["TARGET"]
        X_19, y_19 = m19[cols], m19["TARGET"]
        X_20, y_20 = m20[cols], m20["TARGET"]
        X_21, y_21 = m21[cols], m21["TARGET"]
        X_22, y_22 = m22[cols], m22["TARGET"]
        X_23, y_23 = m23[cols], m23["TARGET"]
        X_24, y_24 = m24[cols], m24["TARGET"]

        for yr, Xyr in zip(range(2013, 2025),
                           [X_13, X_14, X_15, X_16, X_17, X_18,
                            X_19, X_20, X_21, X_22, X_23, X_24]):
            print(len(Xyr), "Samples for -", yr)

        X = pd.concat([
            X_13, X_14, X_15, X_16, X_17, X_18,
            X_19, X_20, X_21, X_22, X_23, X_24
        ])
        y = pd.concat([
            y_13, y_14, y_15, y_16, y_17, y_18,
            y_19, y_20, y_21, y_22, y_23, y_24
        ])

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=cols, index=X.index)

        self.x_train_class, self.x_test_class, self.y_train_class, self.y_test_class = train_test_split(
            X, y, train_size=0.8, stratify=y, random_state=42
        )

        sm = SMOTE(random_state=42)
        self.x_train_class, self.y_train_class = sm.fit_resample(self.x_train_class, self.y_train_class)

    def model_test_classify(self):
        DT = cross_validate(DecisionTreeClassifier(),
                            self.x_train_class, self.y_train_class,
                            cv=10, scoring=["accuracy"], return_train_score=True)
        GB = cross_validate(GradientBoostingClassifier(),
                            self.x_train_class, self.y_train_class,
                            cv=10, scoring=["accuracy"], return_train_score=True)
        SV_C = cross_validate(SVC(kernel="rbf"),
                              self.x_train_class, self.y_train_class,
                              cv=10, scoring=["accuracy"], return_train_score=True)
        RF = cross_validate(RandomForestClassifier(),
                            self.x_train_class, self.y_train_class,
                            cv=10, scoring=["accuracy"], return_train_score=True)
        LR = cross_validate(LogisticRegression(max_iter=1000),
                            self.x_train_class, self.y_train_class,
                            cv=10, scoring=["accuracy"], return_train_score=True)

        print("DT accuracy :", np.mean(DT["test_accuracy"]))
        print("GB accuracy :", np.mean(GB["test_accuracy"]))
        print("SVC accuracy:", np.mean(SV_C["test_accuracy"]))
        print("RF accuracy :", np.mean(RF["test_accuracy"]), "#WINNER")
        print("LR accuracy :", np.mean(LR["test_accuracy"]))

        n_estimators = [int(x) for x in np.linspace(100, 2000, 10)]
        max_features = ["sqrt", "log2", None]
        max_depth = [int(x) for x in np.linspace(1, 110, 10)] + [None]
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        param_grid = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap
        }

        rf = RandomForestClassifier()
        search = RandomizedSearchCV(
            rf, param_grid, n_iter=100, cv=10,
            n_jobs=-1, verbose=2, random_state=42
        )
        search.fit(self.x_train_class, self.y_train_class)
        print("Best parameters:", search.best_params_)

        test_pred = search.predict(self.x_test_class)
        print("Test accuracy :", accuracy_score(self.y_test_class, test_pred))
        return search

    def plot_feature_importance_classify(self, model):
        imp = pd.Series(model.best_estimator_.feature_importances_,
                        index=self.x_test_class.columns).sort_values(ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x=imp, y=imp.index, ax=ax)
        ax.set_title("Random Forest Feature Importances", fontsize=18)
        ax.set_xlabel("Importance", fontsize=14)
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="NFL Combine Classifier — binary SOLO‑tackle prediction (2013‑2024)."
    )
    parser.add_argument(
        "--path", type=str, default="",
        help="Folder with NFL_20xx_edit.xlsx and corresponding *-new-data.xlsx files."
    )
    args = parser.parse_args()

    clf = nflCombineClassify(args.path)
    clf.snaps_to_binary()
    best = clf.model_test_classify()
    clf.plot_feature_importance_classify(best)


if __name__ == "__main__":
    main()
