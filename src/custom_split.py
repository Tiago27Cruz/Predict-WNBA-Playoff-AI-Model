from sklearn.model_selection import BaseCrossValidator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

class customSplit(BaseCrossValidator):
    def __init__(self, df, usepca=True):
        self.df = df
        self.usepca = usepca

    def split(self, X, y=None, groups=None):
        years = sorted(self.df['year'].unique())
        for i in range(1, len(years)):
            train_years = years[:i]
            test_year = years[i]
            filtered_df = self.df[self.df["year"].isin(train_years)].reset_index(drop=True)
            target_df = self.df[self.df["year"] == test_year].reset_index(drop=True)

            X_train = filtered_df.drop(columns=["playoff"])
            y_train = filtered_df["playoff"]
            cols = X_train.columns

            pca = PCA(n_components=11)
            scaler = StandardScaler()
            if self.usepca:
                X_train = pca.fit_transform(scaler.fit_transform(X_train))

                pcas = pd.DataFrame(pca.components_, columns=cols)
                sorted_columns = pcas.apply(lambda row: [col for col, _ in sorted(row.items(), key=lambda x: x[1])][:4], axis=1)
                #print(sorted_columns)

            X_test = target_df.drop(columns=["playoff"])
            y_test = target_df["playoff"]
            if self.usepca:
                X_test = pca.transform(scaler.transform(X_test))

            train_idx = filtered_df.index
            test_idx = target_df.index

            print(f"Train indices: {train_idx}, Test indices: {test_idx}")

            assert len(train_idx) == len(X_train), "Mismatch in training indices and X_train size."
            assert len(test_idx) == len(X_test), "Mismatch in test indices and X_test size."

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.df['year'].unique()) - 1