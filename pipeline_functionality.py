import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
# def process_main(filepath):
#     df = read_df(filepath)
#     return df

#Read data off csv file

class preprocess_data():
    def __init__(self,**kwargs):
        self.filepath = kwargs["filepath"]

    def read_df(self,process_complete = False):
        df = pd.read_csv(self.filepath,encoding="ISO-8859-1")
        df.drop('id',axis = 1,inplace=True)
        df = self.rename_columns(df)
        if process_complete != False:
            #drop na
            df = df.dropna(subset=["Feature 0","Feature 1","Feature 10"])
            df = df.reset_index(drop=True)
            oversample = SMOTE()
            df_train,df_label = oversample.fit_resample(df[[i for i in df.columns.tolist() if i!="label"]],df[[i for i in df.columns.tolist() if i=="label"]])
            df_train["labels"] = df_label
            df = df_train
            #remove highly correlated features
            df = df.drop(['Feature 3', 'Feature 4', 'Feature 7','Age','Sex'],axis = 1)
            feature_eng = feature_engineering()
            df = feature_eng.feature0_fit_transform(df)
            df = feature_eng.feature5_fit_transform(df)
            df = feature_eng.feature10_fit_transform(df)
            df = df.drop(["Feature 1","Feature 5","Feature 10"],axis = 1)
            df = pd.concat([df[[i for i in df.columns.tolist() if "_" not in i and i != "labels"]],pd.get_dummies(df[['Feature 1_Cat','Feature 5_Cat','Feature 10_Cat',]]),df[["labels"]]],axis = 1)
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
    def process(self,df):
        df = df.drop(["Feature 1","Feature 5","Feature 10"],axis = 1)
        columns = [i for i in df.columns.tolist() if "Feature" in i]
        columns.append("labels")
        df = df[columns]
        df = pd.concat([df[[i for i in df.columns.tolist() if "_" not in i and i != "labels"]],pd.get_dummies(df[['Feature 1_Cat','Feature 5_Cat','Feature 10_Cat',]]),df[["labels"]]],axis = 1)
        return df   

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
