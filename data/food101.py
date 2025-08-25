from datasets import load_dataset

food101 = load_dataset("ethz/food101", cache_dir='./food101')

print(len(food101['train']))

print(len(food101['validation']))