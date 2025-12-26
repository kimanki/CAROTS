import pandas as pd
df = pd.read_csv('data/itransformer/traffic.csv', nrows=1)
print(df.shape[1])

