import pandas as pd
import os

from analysis.model.neuromarkers.raw.alpha_asymmetries import AlphaAsymmetries
from analysis.model.neuromarkers.raw.alpha_parietal import AlphaParietal
from analysis.model.neuromarkers.raw.beta_fz import BetaFz
from analysis.model.neuromarkers.raw.channels_db import ChannelsDb
from analysis.model.neuromarkers.raw.channels_z import ChannelsZ
from analysis.model.neuromarkers.raw.channels_z_common_baseline import ChannelsZCommonBaseline
from analysis.model.neuromarkers.raw.theta_fz import ThetaFz
from analysis.model.participant import Participant

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
        self.participant_names = self.load_participant_names()
        self.participants = []

    @staticmethod
    def load_table(data_path, table_path, table_name, extension='.xls', sheet_names=None, usecols=None, skiprows=None):
        df = []
        file_path = data_path + table_path + table_name + extension
        try:
            if extension == '.xls':
                df = pd.read_excel(file_path, sheet_name=None, engine='xlrd', usecols=usecols, skiprows=skiprows)
                print("Successfully loaded! ", df)
            elif extension == '.xlsx':
                df = pd.read_excel(file_path, sheet_name=None, engine='openpyxl', usecols=usecols, skiprows=skiprows)
                print("Successfully loaded! ", df)
            else:
                raise Exception("Format not supported! ", extension)
        except IOError:
            print("There was an error opening the file ", file_path)
        return df

    def load_all_tables(self, table_path):
        for name in self.participant_names:
            table_path = name + table_path
            alpha_parietal = self.load_alpha_parietal_table(table_path)
            alpha_asymmetries = self.load_alpha_asymmetries_table(table_path)
            beta_fz = self.load_beta_fz_table(table_path)
            channels_db = self.load_channels_db_table(table_path)
            channels_z = self.load_channels_z_table(table_path)
            channels_z_cb = self.load_channels_z_common_baseline_table(table_path)
            theta_fz = self.load_theta_fz_table(table_path)
            participant = Participant(name)
            participant.set_alpha_parietal(alpha_parietal)
            participant.set_asymmetries_alpha(alpha_asymmetries)
            participant.set_beta_fz(beta_fz)
            participant.set_channels_db(channels_db)
            participant.set_channels_z(channels_z)
            participant.set_channels_z_cb(channels_z_cb)
            participant.set_theta_fz(theta_fz)
            self.participants.append(participant)
            print("Successfully loaded participant " + name + "!")

    def load_alpha_asymmetries_table(self, table_path):
        dataframe = self.load_table(self.data_path, table_path, ALPHA_ASYMMETRIES)
        return AlphaAsymmetries(dataframe)

    def load_alpha_parietal_table(self, table_path):
        dataframe = self.load_table(self.data_path, table_path, ALPHA_PARIETAL)
        return AlphaParietal(dataframe)

    def load_beta_fz_table(self, table_path):
        dataframe = self.load_table(self.data_path, table_path, BETA_FZ)
        return BetaFz(dataframe)

    def load_channels_db_table(self, table_path):
        dataframe = self.load_table(self.data_path, table_path, CHANNELS_DB)
        return ChannelsDb(dataframe)

    def load_channels_z_table(self, table_path):
        dataframe = self.load_table(self.data_path, table_path, CHANNELS_Z)
        return ChannelsZ(dataframe)

    def load_channels_z_common_baseline_table(self, table_path):
        dataframe = self.load_table(self.data_path, table_path, CHANNELS_Z_COMMON_BASELINE)
        return ChannelsZCommonBaseline(dataframe)

    def load_theta_fz_table(self, table_path):
        dataframe = self.load_table(self.data_path, table_path, THETA_FZ)
        return ThetaFz(dataframe)

    def load_participant_names(self):
        dirs = next(os.walk(self.data_path))[1]
        return dirs

    def get_data_path(self):
        return self.data_path

    def set_data_path(self, data_path):
        self.data_path = data_path

    def get_participant_names(self):
        return self.participant_names

    def get_participants(self):
        return self.participants