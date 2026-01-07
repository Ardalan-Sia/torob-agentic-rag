import pandas as pd

print(pd.read_parquet('/home/ardalan-sia/Documents/ML/Project3/Torob-Database/base_products.parquet', engine='pyarrow').head(2).to_dict())