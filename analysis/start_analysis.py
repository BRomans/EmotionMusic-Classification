from analysis.utils.table_loader import TableLoader

data_path = '../data'
table_loader = TableLoader(data_path)
table_loader.load_all_tables("/POWERS/")
participants = table_loader.get_participants()
print("End of program")