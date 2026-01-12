import pandas as pd

class DataManager:
    def __init__(self, path):
        self.df = pd.read_csv(path)

    def get_by_id(self, pid):
        return self.df[self.df["ID"] == pid]

    def filter_age(self, min_age, max_age):
        return self.df[
            (self.df["Age"] >= min_age) &
            (self.df["Age"] <= max_age)
        ]

    def update_patient(self, pid, column, value):
        self.df.loc[self.df["ID"] == pid, column] = value

    def delete_patient(self, pid):
        self.df = self.df[self.df["ID"] != pid]
