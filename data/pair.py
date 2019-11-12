import pandas as pd
import numpy as np
import random

# get_category_names
with open('list_category_cloth.txt', 'r') as f:
    categories = []
    for i, line in enumerate(f.readlines()):
        if i > 1:
            categories.append(line.split(' ')[0])

# get image category map
with open('list_category_img.txt', 'r') as f:
    images = []
    for i, line in enumerate(f.readlines()):
        if i > 1:
            images.append([word.strip() for word in line.split(' ') if len(word) > 0])

#get train, valid, test split
with open('list_eval_partition.txt', 'r') as f:
    images_partition = []
    for i, line in enumerate(f.readlines()):
        if i > 1:
            images_partition.append([word.strip() for word in line.split(' ') if len(word) > 0])


data_df = pd.DataFrame(images, columns=['images', 'category_label'])
partition_df = pd.DataFrame(images_partition, columns=['images', 'dataset'])

data_df['category_label'] = data_df['category_label'].astype(int)
data_df = data_df.merge(partition_df, on='images')

num_classes = 48
train_indices = []
for i in range(0,len(data_df)):
	if(data_df['dataset'][i]=='train'):
		train_indices.append(i)

digit_indices = [[] for y in range(num_classes+1)] 
for i in train_indices:
    digit_indices[data_df['category_label'][i]].append(i)

for i in digit_indices:
	if(len(i)==0):
		digit_indices.remove(i)
num_classes =  len(digit_indices)

def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    eq = []
    label1 = []
    label2 = []
    n = min([len(digit_indices[d]) for d in range(num_classes)])
    print(n) 
    for d in range(num_classes):
        for i in range(n-1):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            label1 += [d]
            label2 += [d]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            label1 += [d]
            label2 += [dn]
            eq += [1, 0]
    return np.array(pairs), np.array(eq), np.array(label1), np.array(label2)

pairs,eq,l1,l2=create_pairs(data_df['images'],digit_indices)
