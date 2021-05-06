import pandas as pd
import os

ALPHA_ASYMMETRIES = 'Asymmetries Alpha band'
ALPHA_PARIETAL = 'AlphaParietal'
BETA_FZ = 'BetaFz'
CHANNELS_DB = 'Channels_dB'
CHANNELS_Z = 'Channels_z'
CHANNELS_Z_COMMON_BASELINE = 'Channels_zCommonBaseline'
THETA_FZ = 'ThetaFz'


class TableLoader:

    def __init__(self, data_path):
        self.set_data_path(data_path)
        self.load_participant_names()

    @staticmethod
    def load_table(table_path, table_name, extension='.xls'):
        df = []
        file_path = table_path + table_name + extension
        try:
            if extension is '.xls':
                df = pd.read_excel(file_path, sheet_name=None, engine='xlrd')
                print("Successfully loaded! ", df)
            elif extension is '.xlsx':
                df = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
                print("Successfully loaded! ", df)
            else:
                raise Exception("Format not supported! ", extension)
        except IOError:
            print("There was an error opening the file ", file_path)
        return df

    def load_alpha_asymmetries_table(self, table_path):
        return self.load_table(table_path, ALPHA_ASYMMETRIES)

    def load_alpha_parietal_table(self, table_path):
        return self.load_table(table_path, ALPHA_PARIETAL)

    def load_beta_fz_table(self, table_path):
        return self.load_table(table_path, BETA_FZ)

    def load_channels_db_table(self, table_path):
        return self.load_table(table_path, CHANNELS_DB)

    def load_channels_z_table(self, table_path):
        return self.load_table(table_path, CHANNELS_Z)

    def load_channels_z_common_baseline_table(self, table_path):
        return self.load_table(table_path, CHANNELS_Z_COMMON_BASELINE)

    def load_theta_fz_table(self, table_path):
        return self.load_table(table_path, THETA_FZ)

    def load_participant_names(self):
        dirs = next(os.walk(self.data_path))[1]
        self.participant_names = dirs

    def get_data_path(self):
        return self.data_path

    def set_data_path(self, data_path):
        self.data_path = data_path

    def get_participant_names(self):
        return self.participant_names