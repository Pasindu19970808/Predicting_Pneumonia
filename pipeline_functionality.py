from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import torch
from torch import nn
from torch.utils.data import DataLoader
from pydantic import BaseModel
from typing import Dict,Tuple,List,Optional
from pydantic import StrictStr,StrictInt
from pydantic import ValidationError
# def process_main(filepath):
#     df = read_df(filepath)
#     return df

#Read data off csv file

class preprocess_data():
    def __init__(self,**kwargs):
        self.filepath = kwargs["filepath"]
        self.scaler = None
    def read_df(self,process_complete = False,test_df = False,test_path = ""):
        if test_df == True:
            self.filepath = test_path
        df = pd.read_csv(self.filepath,encoding="ISO-8859-1")
        df.drop('id',axis = 1,inplace=True)
        df = self.rename_columns(df)
        if process_complete != False:
            #drop na
            if test_df == False:
                df = df.dropna(subset=["Feature 0","Feature 1","Feature 10"])
                df = df.reset_index(drop=True)
                # oversample = SMOTE()
                # df_train,df_label = oversample.fit_resample(df[[i for i in df.columns.tolist() if i!="label"]],df[[i for i in df.columns.tolist() if i=="label"]])
                # df_train["labels"] = df_label
                # df = df_train
                #remove highly correlated features
                # df = df.drop(['Feature 3', 'Feature 4', 'Feature 7','Age','Sex'],axis = 1)

                self.feature_eng = feature_engineering()
                df = self.feature_eng.feature0_fit_transform(df)
                df = self.feature_eng.feature5_fit_transform(df)
                df = self.feature_eng.feature10_fit_transform(df)
            else:
                df = self.feature_eng.transform_df(df)
            df.loc[df["Sex"] == 1,"Sex"] = "M"
            df.loc[df["Sex"] == 0,"Sex"] = "F"
            #works for training df and not test df
            try:
                df = pd.concat([df[[i for i in df.columns.tolist() if "_" not in i and i != "labels"]],pd.get_dummies(df[['Feature 1_Cat','Feature 5_Cat','Feature 10_Cat',"Sex"]]),df[["labels"]]],axis = 1)
            except:
                #works for test df
                df = pd.concat([df[[i for i in df.columns.tolist() if "_" not in i and i != "labels"]],pd.get_dummies(df[['Feature 1_Cat','Feature 5_Cat','Feature 10_Cat',"Sex"]])],axis = 1)
            df = df.drop(["Feature 1","Feature 5","Feature 10","Sex"],axis = 1)

            if test_df == False:
                self.scaler = StandardScaler()
                self.scaler.fit(df[["Feature 0","Feature 2","Feature 6",'Feature 3', 'Feature 4', 'Feature 7','Age']])
            df[["Feature 0","Feature 2","Feature 6",'Feature 3', 'Feature 4', 'Feature 7','Age']] = self.scaler.transform(df[["Feature 0","Feature 2","Feature 6",'Feature 3', 'Feature 4', 'Feature 7','Age']])
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
            self.feature1boundary = df.loc[df["label"] == 0]["Feature 1"].describe()["75%"] + 1.5*(df.loc[df["label"] == 0]["Feature 1"].describe()["75%"] - df.loc[df["label"] == 0]["Feature 1"].describe()["25%"])
            df = self.apply_change(df,"Feature 1",{"LTE":"Low Risk","GT":"High Risk"},self.feature1boundary)
            return df
        else:
            raise Warning("Feature 0 boundary to split categorical variable is already set. No change to the data was done")
    def feature5_fit_transform(self,df):
        if (self.feature5boundary == None) and ("Feature5_Cat" not in df.columns.tolist()):
            self.feature5boundary = df.loc[df["label"] == 1]["Feature 5"].describe()["75%"]
            df = self.apply_change(df,"Feature 5",{"LTE":"High Risk","GT":"Low Risk"},self.feature5boundary)
            # df.loc[df["Feature 5"] <= self.feature5boundary,'Feature5_Cat'] = "High Risk"
            # df.loc[df["Feature 5"] > self.feature5boundary,'Feature5_Cat'] = "Low Risk"
            return df
        else:
            raise Warning("Feature 5 boundary to split categorical variable is already set. No change to the data was done")
    def feature10_fit_transform(self,df):
        if (self.feature10boundary == None) and ("Feature10_Cat" not in df.columns.tolist()):
            self.feature10boundary = df.loc[df["label"] == 0]["Feature 10"].describe()["75%"]
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
        return df
    def apply_change(self,df,col,change_dict,boundary):
        catcol = col + "_Cat"
        df[catcol] = np.nan
        df.loc[df[col] <= boundary,catcol] = change_dict["LTE"]
        df.loc[df[col] > boundary,catcol] = change_dict["GT"]
        return df


#Do validation of the tuple carrying the layer details
class Type_Checks(BaseModel):
    layer_info_dict : Dict[StrictInt,Tuple[StrictInt,StrictInt,StrictStr]]



class NeuralNetwork(nn.Module):
    def __init__(self,layer_info_dict):
        """
        - Takes in a dictionary where key is the layer number and the tuples are (input_shape,number_of_hidden_units_on_layer)
        - The values are tuples of the form (input shape(int), number of hidden units in layer(int),activation function to use(string))
        - If an activation layer is not required
        """
        self.activation_layers = {
        'relu':nn.ReLU(),
        'sigmoid':nn.Sigmoid(),
        'tanh':nn.Tanh(),
        'elu':nn.ELU(),
        }
        try:
            Type_Checks(layer_info_dict = layer_info_dict)
            super(NeuralNetwork,self).__init__()
            layers = []
            for layer in layer_info_dict:
                layers.append(nn.Linear(in_features=layer_info_dict[layer][0],out_features=layer_info_dict[layer][1],bias = True))
                if (layer_info_dict[layer][2] != None) and (layer_info_dict[layer][2] in list(self.activation_layers.keys())):
                    layers.append(self.activation_layers[layer_info_dict[layer][2]])
                else:
                    raise Exception("Invalid Activation Layer name")
            self.linear_stack = nn.Sequential(*layers)
        except ValidationError as e:
            raise Exception(e.json())
    def forward(self,x):
        forward_prop = self.linear_stack(x)
        return forward_prop



class TrainingValidatingNetwork:
    def __init__(self):
        pass
    def train(self,dataloader,model,loss_fn,optimizer):
        model.train()
        for batch,(X,Y) in enumerate(dataloader):
            pred = model(X.float())
            loss = loss_fn(pred,Y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    def validate(self,dataloader,model,loss_fn,test = False):
        model.eval()
        testing_loss = 0
        correct = 0
        validate_inst_count = 0
        with torch.no_grad():
            for batch,(X,Y) in enumerate(dataloader):
                y_pred = model(X.float())
                loss = loss_fn(y_pred,Y.float())
                testing_loss += loss.item()
                validate_inst_count += 1
                #y_pred.argmax(1) == y will return True or False
                #as it is being tested one instance at a time,
                #it will be one True or False
                #.astype(torch.float) makes it to a 1 or 0
                #As its one instance at a time, no need to sum
                #output y_pred is of shape [1], y is of shape[]
                correct += (y_pred.round() == Y.float()).type(torch.float).item()
        #calculate average loss across the batches
        avg_loss = testing_loss/validate_inst_count
        #calculate average accuracy across the batches
        avg_accuracy = correct/validate_inst_count
        return avg_accuracy,avg_loss,model
    def train_and_validate(self,train_dataloader,validate_dataloader,model,loss_fn,epochs,optimizer,predicting = False):
        self.optimizer = optimizer
        for i in range(epochs):
            self.train(train_dataloader,model,loss_fn,self.optimizer)
        if predicting == False:
            avg_accuracy,avg_loss,model = self.validate(validate_dataloader,model,loss_fn)
            return avg_accuracy,avg_loss,model
        else:
            #If we are only trying to predict for Submission purposes, no validation will be done
            return model








