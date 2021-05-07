
class AlphaParietal:

    def __init__(self, dataframe):
        self.z_score_p4 = dataframe['Z-scoreP4']
        self.db_p4 = dataframe['DbP4']
        self.z_score_p3 = dataframe['Z-scoreP3']
        self.db_p3 = dataframe['DbP3']
