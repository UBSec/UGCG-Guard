import pandas as pd
from sklearn.metrics import classification_report

df = pd.read_csv("ugcg_gpt.csv")
df['prediction'] = -99
for index, row in df.iterrows():
    label = row['gpt_output'].lower()
    if 'unsafe' in label:
        df.at[index, 'prediction'] = 1
    elif 'safe' in label:
        df.at[index, 'prediction'] = 0
    else:
        df.at[index, 'prediction'] = -1
        print("Error: ", row['img_path'], row['gpt_output']) # verify if there are any errors

print(df.groupby('prediction').count())
report = classification_report(df['label'], df['prediction'], target_names=['safe', 'unsafe'])

print("Classification Report: \n", report)