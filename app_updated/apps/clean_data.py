import pandas as pd
import numpy as np

def clean_data(df):
    df = df.replace('?',np.nan)
    df_isna = pd.isna(df)
    num_rows_with_excessive_na = np.sum(np.sum(df_isna,axis=1)>(df_isna.shape[1]/2))
    excessive_na_row_percentage = num_rows_with_excessive_na/df_isna.shape[0]
    num_fields_with_excessive_na = np.sum(np.sum(df_isna,axis=0)>(df_isna.shape[0]/2))
    excessive_na_field_percentage = num_fields_with_excessive_na/df_isna.shape[1]
    num_fields_with_excessive_na
    field_contain_na = np.sum(df_isna,axis=0)>(df_isna.shape[1]/2)
    field_key = field_contain_na.keys()
    for each_key in ["iyear","gname","nkill","nwound","weaptype1_txt","nperps","individual","nkillus"]:
        if each_key in field_key:
            field_contain_na[each_key]=False
    df_drop_na_field = df.loc[:,field_contain_na!=True]
    fields_with_na = df_drop_na_field.columns[(np.sum(pd.isna(df_drop_na_field),axis=0)>0)]
    for each_col in fields_with_na:
        df_drop_na_field.loc[pd.isna(df_drop_na_field)[each_col],each_col] = round(np.mean(df_drop_na_field.loc[pd.isna(df_drop_na_field)[each_col]!=True,each_col]))
    df = df_drop_na_field
    return df