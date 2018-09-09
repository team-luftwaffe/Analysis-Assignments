# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:41:29 2018

@author: ajayku
"""

#Importing Required libraries
from sys import argv
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from  datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,classification_report
from sklearn.feature_selection import RFE,RFECV
from sklearn import svm
import seaborn as sns

class wine:
    
    def __init__(self):
        pass
    
    def Preprocessing(self,wine_df):
        wine_df_cleaned = wine_df.fillna(wine_df.median())
        
        #Convert production date column to datetiem format
        wine_df_cleaned['production date']=pd.to_datetime(wine_df_cleaned['production date'],
                                           infer_datetime_format=True)
        #Getting the current date
        current_date=datetime.now()
        
        # Extracting a new feature age_of_wine by calculating the days from the production date and current date
        deltas=[]
        for i in range(len(wine_df_cleaned)):
            delta=current_date-wine_df_cleaned['production date'][i]
            deltas.append(delta.days)
        wine_df_cleaned['age_of_wine']=deltas
        
        # Use the 1.5* IQR technique to remove all the outlier points from the data set
        wine_df_cleaned_p=wine_df_cleaned.iloc[:,1:13]
        
        # For each feature find the data points with extreme high or low values
        for feature in wine_df_cleaned_p.keys():

            # TODO: Calculate Q1 (25th percentile of the data) for the given feature
            Q1 = np.percentile(wine_df_cleaned_p[feature], q=25)
 
            # TODO: Calculate Q3 (75th percentile of the data) for the given feature
            Q3 = np.percentile(wine_df_cleaned_p[feature], q=75)
 
            # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
            interquartile_range = Q3 - Q1
            step = 1.5 * interquartile_range
 
            # Remove the outliers
            wine_df_cleaned_p=wine_df_cleaned_p[((wine_df_cleaned_p[feature] >= Q1 - step) 
                    & (wine_df_cleaned_p[feature] <= Q3 + step))]
        
        return wine_df_cleaned_p
    
    def FeatureSelection(self,wine_df_cleaned_p,bin=1):
        if bin==1:
            # Splitting the data into train and test data
            Y_wine=wine_df_cleaned_p['quality']
            X_wine=wine_df_cleaned_p.drop(['quality'],axis=1)
        else:
            # Split the data into features and target label
            Y_wine = wine_df_cleaned_p['quality_category']
            X_wine= wine_df_cleaned_p.drop(['quality', 'quality_category'], axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X_wine, Y_wine,
                                           test_size=0.30, random_state=123)

        # Normalizing the data
        std_scale = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(X_train)
        X_train = std_scale.transform(X_train)
        X_test = std_scale.transform(X_test)
        base_model=RandomForestClassifier(random_state=0)
        base_model.fit(X_train,y_train)
        rfe = RFE(base_model, 7)
        rfe_fit = rfe.fit(X_train, y_train)
        #print("Num Features:",rfe_fit.n_features_)
        #print("Selected Features: ",rfe_fit.support_)
        #print("Feature Ranking: ",rfe_fit.ranking_)
        X_train_rfe=rfe_fit.transform(X_train)
        X_test_rfe=rfe_fit.transform(X_test)
        return X_train_rfe,X_test_rfe,y_train,y_test
        
    def CategorizeBin(self,wine_df_cleaned_p):
        #Defining the splits for categories. 1–4 will be bad quality, 5–6 will be average, 7–10 will be best
        bins = [1,4,6,10]
        #0 for low quality, 1 for average, 2 for great quality
        quality_labels=[0,1,2]
        wine_df_cleaned_p['quality_category'] = pd.cut(wine_df_cleaned_p['quality'], bins=bins, labels=quality_labels, include_lowest=True)
        return wine_df_cleaned_p
    
    def RandomForestModel(self,X_train,y_train,X_test,y_test):
        RF=RandomForestClassifier(random_state=0)
        RF.fit(X_train,y_train)
        y_predict=RF.predict(X_test)
        self.PerformanceMetrics(y_test,y_predict,'RandomForest')
        
    def LogisticRegressionModel(self,X_train,y_train,X_test,y_test):
        log=linear_model.LogisticRegression(multi_class='multinomial',solver='newton-cg').fit(X_train,y_train)
        # Predicting the test data
        y_predict=log.predict(X_test)
        self.PerformanceMetrics(y_test,y_predict,'LogisticRegression')
        
    def SupportVectorMachineModel(self,X_train,y_train,X_test,y_test):
        #Using SVM for classification
        svm_clf = svm.SVC(C=1.0)
        svm_clf.fit(X_train, y_train)
        y_predict=svm_clf.predict(X_test)
        self.PerformanceMetrics(y_test,y_predict,'SVM')
        
    def PerformanceMetrics(self,y_test,y_predict,algo):    
        #Performance metrics for the above model with rfe selected the features
        print('\n')
        print(algo)
        RF_acc=accuracy_score(y_test,y_predict)
        print('Accuracy is: ',RF_acc)
        RF_cm = confusion_matrix(y_test,y_predict)
        print(RF_cm)
        RF_cr= classification_report(y_test,y_predict)
        print(RF_cr)

def main():
    csv_path = argv[1]
    method=int(argv[2])
    algorithm=int(argv[3])
    wine_object=wine()
    wine_df = pd.read_csv(csv_path)
    wine_df_p=wine_object.Preprocessing(wine_df)
    if method==1:
        X_train_rfe,X_test_rfe,y_train_rfe,y_test_rfe=wine_object.FeatureSelection(wine_df_p,method)
    else:
        wine_df_p=wine_object.CategorizeBin(wine_df_p)
        X_train_rfe,X_test_rfe,y_train_rfe,y_test_rfe=wine_object.FeatureSelection(wine_df_p,method)
    
    if algorithm==1:
        wine_object.RandomForestModel(X_train_rfe,y_train_rfe,X_test_rfe,y_test_rfe)
    elif algorithm==2:
        wine_object.LogisticRegressionModel(X_train_rfe,y_train_rfe,X_test_rfe,y_test_rfe)
    elif algorithm==3:
        wine_object.SupportVectorMachineModel(X_train_rfe,y_train_rfe,X_test_rfe,y_test_rfe)
    
if __name__ == "__main__":       
    main()
    