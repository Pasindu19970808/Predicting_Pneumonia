import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# def process_main(filepath):
#     df = read_df(filepath)
#     return df

#Read data off csv file

class preprocess_data():
    def __init__(self,**kwargs):
        self.filepath = kwargs["filepath"]

    def read_df(self):
        df = pd.read_csv(self.filepath,encoding="ISO-8859-1")
        df.drop('id',axis = 1,inplace=True)
        return df
        # if split == True:
        #     df_labels = df.loc[:,"label"]
        #     df_train = df.loc[:,[i for i in df.columns.tolist() if i != "label"]]
        #     return df_train,df_labels
        # else:
            


    #Rename columns for easier reading
    def rename_columns(self,df:pd.DataFrame):
        self.mapped_dict = self.__mapped_col_names(df)
        df.rename(self.mapped_dict,axis=1,inplace=True)
        return df

    #Map colnames to names we require
    def __mapped_col_names(self,df:pd.DataFrame):
        mapped_dict = {j:"Feature " + str(i) for i,j in zip(range(len(df.columns.tolist())),df.columns.tolist()) if j not in ["Age","Sex 0M1F","label"]}
        mapped_dict["Age"] = "Age"
        mapped_dict["Sex 0M1F"] = "Sex" 
        mapped_dict["label"] = "label"
        return mapped_dict

# def normalize_dat

class preprocess_training_validation:
    def __init__(self):
        self.scaler = None
    def normalize(self,df):
        cols = [i for i in df.columns.tolist() if i != "Age" and i != "Sex"]
        age_sex = df[["Age","Sex"]]
        if self.scaler == None:
            self.scaler = StandardScaler()
            self.scaler.fit(df[cols])
            scaled_data = self.scaler.transform(df[cols])
            scaled_df = pd.DataFrame(scaled_data,columns=cols)
            scaled_df[["Age","Sex"]] = age_sex
            return scaled_df
        else:
            scaled_data = self.scaler.transform(df[cols])
            scaled_df = pd.DataFrame(scaled_data,columns=cols)
            scaled_df[["Age","Sex"]] = age_sex
            return scaled_df
            


