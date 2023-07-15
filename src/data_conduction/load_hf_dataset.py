import os
from datasets import load_dataset
dataset_name = 'bigcode/the-stack-dedup' 

ds = load_dataset(
    path=dataset_name,
    data_dir='data/python',
    save_infos=True,
    split='train',
    num_proc=8
)
print(ds)
split_datasets = ds.train_test_split(test_size=.005)
print(split_datasets)

test_dataset = split_datasets['test']
test_dataset.to_csv('data/real_datasets/thestack-python-val.csv')
print("Test Dataset Done")
print("Process Done")



