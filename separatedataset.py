

import pandas as pd
dataset = pd.read_csv('../data/train_dataset.csv')


# Separate the general dataset into different csv files by PlantName
for region, plant in dataset.groupby('PlantName'):
    plant.to_csv('{}.csv'.format(region), header=True, index_label=True)