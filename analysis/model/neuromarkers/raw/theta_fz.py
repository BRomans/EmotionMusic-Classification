
class ThetaFz:

    def __init__(self, dataframe):
        self.z_score = dataframe['Z-score']
        self.db = dataframe['Db']
        self.theta_idx = dataframe['NEW']