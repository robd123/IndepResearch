import pandas as pd
from pathlib import Path

df_fake = pd.read_csv('D:\\nlp_data_scripts\\fake.csv')
df_fake = df_fake[['text', 'type']]
df_fake = df_fake.rename(columns={'type':'label'})
#print(df_fake.label.value_counts())

df_train = pd.read_csv('D:\\nlp_data_scripts\\train.csv')
df_train = df_train[['text', 'label']]
df_train = df_train[df_train['label'] == 0]
df_train['label'] = df_train['label'].astype(str)
df_train['label'] = df_train['label'].replace('0', 'reliable')
#print(df_train.label.value_counts())

df_full = df_fake.append(df_train)

df_full['num_words'] = df_full['text'].str.split().str.len()
df_full = df_full[df_full['num_words'] > 120]
df_full = df_full[['text', 'label']]
print(df_full.shape)
print(df_full.label.value_counts())
print(df_full.isna().sum())

filepath = Path('D:\\nlp_data_scripts\\fake_full.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
df_full.to_csv(filepath, index = False)