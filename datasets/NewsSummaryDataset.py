import pandas as pd

df = pd.read_csv("../data/news_summary.csv", encoding="latin1")
df.drop_duplicates(subset=["ctext"], inplace=True)
df.dropna(inplace=True)
df.drop(['author', 'date', 'headlines', 'read_more'], 1, inplace=True)
df.reset_index(drop=True,inplace=True)
print(df.describe())