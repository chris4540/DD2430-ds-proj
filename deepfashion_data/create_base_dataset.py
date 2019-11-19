def create_base_dataset():

    df_cat_img = pd.read_csv('list_category_img.txt', sep='\s+', skiprows=1)
    df_eval_partn = pd.read_csv('list_eval_partition.txt', sep='\s+', skiprows=1)
    
    df_cat_img_partn = df_cat_img.merge(df_eval_partn, on='image_name')
    
    df_cat_img_partn_train = df_cat_img_partn[df_cat_img_partn.evaluation_status=='train']
    df_cat_img_partn_train_sorted = df_cat_img_partn_train.sort_values('category_label')
    
    return df_cat_img_partn_train_sorted
    