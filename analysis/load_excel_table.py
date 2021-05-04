import pandas as pd

def load_table(path):
    df = pd.read_excel(path, sheet_name=None, engine='xlrd')
    print(df)
    return df



df = load_table("../data/AEILI/POWERS/Asymmetries Alpha band.xls")
print("Done")


