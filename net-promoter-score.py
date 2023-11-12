import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.DataFrame(np.random.randint(9, 11, size=(1000, 1)), columns=['Would you recommend the product?'])
df2 = pd.DataFrame(np.random.randint(7, 9, size=(400, 1)), columns=['Would you recommend the product?'])
df3 = pd.DataFrame(np.random.randint(0, 7, size=(100, 1)), columns=['Would you recommend the product?'])

df = pd.concat([df1, df2, df3], ignore_index=True)

country_origin_map = {1: 'Poland', 2: 'Germany', 3: 'Netherlands', 4: 'Belgium', 5: 'Denmark'}
country_destination_map = {1: 'Spain', 2: 'Italy'}

df['country_origin'] = np.random.choice(list(country_origin_map.values()), df.shape[0])
df['country_destination'] = np.random.choice(list(country_destination_map.values()), df.shape[0])


def calculate_nps(scores):
    promoters = (scores == 9) | (scores == 10)
    detractors = scores < 7
    nps = ((promoters.sum() - detractors.sum()) / len(scores)) * 100
    return nps


nps_scores = df.groupby(['country_origin', 'country_destination']).apply(
    lambda group: calculate_nps(group['Would you recommend the product?'])
).reset_index()

nps_scores.rename(columns={0: 'NPS'}, inplace=True)

highest_nps = nps_scores.loc[nps_scores.groupby('country_origin')['NPS'].idxmax()]

print(highest_nps)

sns.set_style('whitegrid')

plt.figure(figsize=(14, 7))

sns.barplot(
    x='NPS',
    y='country_origin',
    hue='country_destination',
    data=nps_scores,
    palette='muted'
)

plt.title('NPS Scores by Country Origin and Destination')
plt.xlabel('Net Promoter Score (NPS)')
plt.ylabel('Country Origin')

plt.show()
