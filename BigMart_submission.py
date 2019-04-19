# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 19:01:43 2018

@author: nmadapati
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn import linear_model 
from sklearn import metrics 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score
from sklearn.metrics import  classification_report
from sklearn.metrics import  confusion_matrix




####################################################################
# Preparig  data 
#
##########################################################################

def cleaingData(data) :
    
    ####################################################################
    #Cleaning data 
    #
    ##########################################################################

    data['Item_Fat_Content'] = data['Item_Fat_Content'].apply(
                                      lambda x: "Low Fat" if x == "low fat" else x)
    data['Item_Fat_Content'] = data['Item_Fat_Content'].apply(
                                      lambda x: "Regular" if x == "reg" else x)
    data['Item_Fat_Content'] = data['Item_Fat_Content'].apply(
                                     lambda x: "Low Fat" if x == "LF" else x)
    
    wight_item_type = data[['Item_Weight','Item_Type']].groupby('Item_Type').mean()
    
    item_weight_mean = wight_item_type.mean()
    
    #data['Item_Visibility'] = data['Item_Visibility'].apply(lambda x: x*x )
     
    item_Visibility_mean = data['Item_Visibility'].mean()
    
    data['Item_Weight'] = data['Item_Weight'].apply(
                     lambda x: float(item_weight_mean.values) if pd.isnull(x) else x)
    
    data['Outlet_Size'] = data['Outlet_Size'].apply(
                     lambda x: "Medium" if pd.isnull(x) else x)
    
    data['Item_Visibility'] = data['Item_Visibility'].apply(
                    lambda x: item_Visibility_mean if x==0 else x)
    
    
    ###########################################################
    #### Converting  categorical variable into dummy variables
    ################################################################
    
    itemType = pd.get_dummies(data['Item_Type'], drop_first=True)
    itemFat = pd.get_dummies(data['Item_Fat_Content'], drop_first=True)
    outlettype  = pd.get_dummies(data['Outlet_Type'], drop_first =True)
    outLoctionType  = pd.get_dummies(data['Outlet_Location_Type'], drop_first =True)
    store_size  = pd.get_dummies(data['Outlet_Size'], drop_first =True)
    
    data = pd.concat([data,  itemType, itemFat, store_size, outlettype], axis =1)
    
    ###################################################################
    ### Droping categorical  variabla and other other obvious data
    #####################################################################
    
    drop_items = ['Item_Identifier','Outlet_Identifier', 'Outlet_Size',
                 'Item_Type', 'Item_Fat_Content', 
                'Outlet_Type', 'Outlet_Location_Type']
               
    
    data.drop(drop_items, axis=1, inplace= True)
    
    return data
 
def vif_selection(data) :
    ###################################################################
    ### Calculating VIF to identify significant  dependent vaiable 
    #####################################################################
    
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    vif["features"] =data.columns
    
 
    ###################################################################
    ### Droping variable based on VIF data 
    #####################################################################
    
    data.drop(['Outlet_Establishment_Year'], axis =1, inplace= True )
    
       
    return data



####################################################
# Read Data 
###################################################
input_data_raw = pd.read_csv("Train_UWu5bXk.csv")  
input_expected_Data = input_data_raw['Item_Outlet_Sales']
input_data = input_data_raw.drop('Item_Outlet_Sales', axis =1 )  
input_data = cleaingData(input_data)
input_data = vif_selection(input_data)

###################################################################
### Linear  regression 
#####################################################################
def linear_reg_model(data) :
#x = CalculatingPCA(input_data)
    x= input_data
    y=input_data_raw['Item_Outlet_Sales']
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.33, random_state=40)
    #print(X_train.head())
    ln = linear_model.LinearRegression()
    ln.fit(X_train,Y_train)
    #print(ln.coef_)
    #coef_df = pd.DataFrame(ln.coef_, index=x.columns, columns=['Coefficient'] )
   
    predict_sales = ln.predict(X_test)
        
        #sns.distplot(Y_test-predict_sales)
    print("R^2 sqaure ", metrics.r2_score(Y_test,predict_sales))
    print('MAE:', metrics.mean_absolute_error(Y_test,predict_sales))
    print('MSE:', metrics.mean_squared_error(Y_test,predict_sales))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test,predict_sales)))
      
   
    return ln.predict(data)
    
    
    


#####################################################################
## Preparing test data 
#####################################################################
test_data = pd.read_csv("Test_u94Q5KV.csv")    
test_data = cleaingData(test_data)
test_data = vif_selection(test_data)

###################################################################
### Call regression model and write to file
###################################################################
predict_sales = linear_reg_model(test_data)
df = pd.DataFrame(predict_sales)
df.to_csv("output2.csv")


