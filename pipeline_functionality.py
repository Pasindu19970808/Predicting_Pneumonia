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
            

class feature_engineering:
    def __init__(self):
        self.feature1boundary = None
        self.feature5boundary = None
        self.feature10boundary = None
    def feature0_fit_transform(self,df):
        if (self.feature1boundary == None) and ("Feature1_Cat" not in df.columns.tolist()):
            self.feature1boundary = df.loc[df["labels"] == 0]["Feature 1"].describe()["75%"] + 1.5*(df.loc[df["labels"] == 0]["Feature 1"].describe()["75%"] - df.loc[df["labels"] == 0]["Feature 1"].describe()["25%"])
            df = self.apply_change(df,"Feature 1",{"LTE":"Low Risk","GT":"High Risk"},self.feature1boundary)
            return df
        else:
            raise Warning("Feature 0 boundary to split categorical variable is already set. No change to the data was done")
    def feature5_fit_transform(self,df):
        if (self.feature5boundary == None) and ("Feature5_Cat" not in df.columns.tolist()):
            self.feature5boundary = df.loc[df["labels"] == 1]["Feature 5"].describe()["75%"]
            df = self.apply_change(df,"Feature 5",{"LTE":"High Risk","GT":"Low Risk"},self.feature5boundary)
            # df.loc[df["Feature 5"] <= self.feature5boundary,'Feature5_Cat'] = "High Risk"
            # df.loc[df["Feature 5"] > self.feature5boundary,'Feature5_Cat'] = "Low Risk"
            return df
        else:
            raise Warning("Feature 5 boundary to split categorical variable is already set. No change to the data was done")
    def feature10_fit_transform(self,df):
        if (self.feature10boundary == None) and ("Feature10_Cat" not in df.columns.tolist()):
            self.feature10boundary = df.loc[df["labels"] == 0]["Feature 10"].describe()["75%"]
            df = self.apply_change(df,"Feature 10",{"LTE":"Low Risk","GT":"High Risk"},self.feature10boundary)
            # df.loc[df["Feature 10"] <= self.feature10boundary,'Feature10_Cat'] = "Low Risk"
            # df.loc[df["Feature 10"] > self.feature10boundary,'Feature10_Cat'] = "High Risk"
            return df
        else:
            raise Warning("Feature 10 boundary to split categorical variable is already set. No change to the data was done")
    #This function is used to transform the final dataframe to be used for cross validation and training
    def transform_df(self,df):
        df = self.apply_change(df,"Feature 1",{"LTE":"Low Risk","GT":"High Risk"},self.feature1boundary)
        df = self.apply_change(df,"Feature 5",{"LTE":"High Risk","GT":"Low Risk"},self.feature5boundary)
        df = self.apply_change(df,"Feature 10",{"LTE":"Low Risk","GT":"High Risk"},self.feature10boundary)

    def apply_change(self,df,col,change_dict,boundary):
        catcol = col + "_Cat"
        df[catcol] = np.nan
        df.loc[df[col] <= boundary,catcol] = change_dict["LTE"]
        df.loc[df[col] > boundary,catcol] = change_dict["GT"]
        return df