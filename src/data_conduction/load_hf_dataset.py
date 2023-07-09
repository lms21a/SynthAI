from datasets import load_dataset
from itertools import islice 
ds = load_dataset("bigcode/the-stack-dedup", data_dir ='data/python', split = 'train')

num_iterations = 0
max_iterations = 10000 * 100

for index, sample in islice(enumerate(ds),max_iterations): num_iterations +=1 

print(num_iterations)
