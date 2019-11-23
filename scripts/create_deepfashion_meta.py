#!/usr/bin/env python
"""
                          Script documentation
Target:
    Generate
        - full data metadata csv
        - sampled data metadata csv

Usage:
    scripts/create_deepfashion_meta.py

Dataset webpage:
    http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html
------------------------------------------------------------------------------
"""
from os.path import join as path_join
import pandas as pd
# ------------------------------
class Config:
    data_root = "deepfashion_data"
    fulldata_csv_fname = 'full_data_meta.csv'

    # --------------------
    # resampling config
    # --------------------
    min_nb_cnt = 5000
    rnd_seed = 42
    sampled_data_csv_fname = 'sampled_data_meta.csv'
# ------------------------------


if __name__ == '__main__':
    # alias
    cfg = Config

    ########################################
    # Load data from the plain text files
    ########################################
    # get_category_names
    file = path_join(cfg.data_root, "list_category_cloth.txt")
    with open(file, 'r') as f:
        categories = []
        for i, line in enumerate(f.readlines()):
            if i > 1:
                categories.append(line.split(' ')[0])

    # get image category map
    file = path_join(cfg.data_root, "list_category_img.txt")
    with open(file, 'r') as f:
        images = []
        for i, line in enumerate(f.readlines()):
            if i > 1:
                images.append([word.strip() for word in line.split(' ') if len(word) > 0])

    #get train, valid, test split
    file = path_join(cfg.data_root, "list_eval_partition.txt")
    with open(file, 'r') as f:
        images_partition = []
        for i, line in enumerate(f.readlines()):
            if i > 1:
                images_partition.append([word.strip() for word in line.split(' ') if len(word) > 0])
    # ---------------------------------------------------------------------------------------------------
    ############################################################
    # Construct the pandas dataframe with the data loaded
    ############################################################
    img_df = pd.DataFrame(images, columns=['images', 'category_label'])
    partition_df = pd.DataFrame(images_partition, columns=['images', 'dataset'])

    # category_label: str -> int
    img_df = img_df.astype({'category_label': 'int32'})

    data_df = img_df.merge(partition_df, on='images')
    data_df['category'] = data_df['category_label'].apply(lambda x: categories[x - 1])

    print("---------- Metadata Summary ----------")
    print(data_df['dataset'].value_counts())
    nclasses = data_df['category'].nunique()
    print("# of classes: ", nclasses)
    print("--------------------------------------")
    #
    classes = data_df['category'].unique()
    classes.sort()

    # build a mapping from category to index
    cat_to_idx = {k:i for i, k in enumerate(classes)}
    # map category_label to label (index)
    data_df['label'] = data_df['category'].map(cat_to_idx)
    # ---------------------------------------------------------------------------------
    # save the full data as  csv
    file = path_join(cfg.data_root, cfg.fulldata_csv_fname)
    print("Saving {} ....".format(file))
    data_df.to_csv(file, index=False)
    # ---------------------------------------------------------------------------------
    ############################################################
    # Sub-sampling
    ############################################################
    col = 'category'
    # filter the categories of which the number is less than the min_nb_cnt
    filtered_df = data_df.groupby(col).filter(lambda x : len(x) > cfg.min_nb_cnt)
    # select the number of data to be sampled
    n_sample_data = filtered_df[col].value_counts().min()
    # re-sample of the column
    sampled_df = \
        (filtered_df.groupby(col)
            .apply(lambda c: c.sample(n_sample_data, random_state=cfg.rnd_seed))
            .reset_index(drop=True))
    print("------ Sampled Metadata Summary ------")
    print(sampled_df[col].value_counts())
    print("--------------------------------------")
    # save the sampled data as  csv
    file = path_join(cfg.data_root, cfg.sampled_data_csv_fname)
    print("Saving {} ....".format(file))
    sampled_df.to_csv(file, index=False)
