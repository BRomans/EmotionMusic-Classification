import pandas as pd
import os


class TableLoader:

    def __init__(self, data_path):
        self.set_data_path(data_path)
        self.load_participant_names()

    @staticmethod
    def load_table(table_path, table_name):
        df = []
        file_path = table_path + table_name
        try:
            df = pd.read_excel(file_path , sheet_name=None, engine='xlrd')
            print("Successfully loaded! ", df)
        except IOError:
            print("There was an error opening the file ", file_path)
        return df

    def load_participant_names(self):
        dirs= next(os.walk(self.data_path))[1]
        self.participant_names = dirs

    def get_data_path(self):
        return self.data_path

    def set_data_path(self, data_path):
        self.data_path = data_path

    def get_participant_names(self):
        return self.participant_names