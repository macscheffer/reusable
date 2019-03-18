
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random


def macs_pca_features(df, cols, pcs, clusters, prefix):
     
     #add a function/functionality to 
     #vizualize sufficient pcs

    
     # takes in a df. columns to reduce   dimensions, how many dimensions to reduce to, and how many clusters to make.
     
     
     df = df.copy()

     scaler = StandardScaler()
     std_df = scaler.fit_transform(df[cols])
     
     pca = PCA(pcs)
     pc_df = pca.fit_transform(std_df)
     

     for cluster in clusters:
         km = KMeans(cluster)
         km = km.fit(pc_df)
         df[prefix + str(pcs) + ‘PCs_’ + str(cluster) + ‘Means_Cluster] = km.labels_

     return df

##################################################################################################################################

def date_to_features(col, df=df, drop=False):
    
    # a function to split a pandas datetime column into a different date columns (year, month, day, day of week, day of year)
     datetime = pd.to_datetime(df[col])
     df['datetime_year'] = datetime.dt.year
     df['datetime_month'] = datetime.dt.month
     df['datetime_dayofweek'] = datetime.dt.dayofweek
    
     df['datetime_dayofyear'] = datetime.dt.dayofyear
     df['datetime_quarter'] = datetime.dt.quarter
     df['datetime_weekofyear'] = datetime.dt.weekofyear
    
     if drop == True:
          return df.drop(columns=col)
     return df

##################################################################################################################################

def macs_labelEncode_meaningful(df, train, cols, target):
    
     # takes in a list of columns to encode in a linear fashion. 
     
     for col in cols:
        
          cur_encodes = pd.factorize(train.pivot_table(index=col, values=target).sort_values(target).index)[1].tolist()
          fnl_encodes = pd.factorize(train.pivot_table(index=col, values=target).sort_values(target).index)[0].tolist()
        
          mapper_dic = dict(zip(cur_encodes, fnl_encodes))
        
          df['MeaningfulEnc_' + col] = df[col].map(mapper_dic)
        
     return df


##################################################################################################################################

def classifers_evaluator(df, base_features, features_to_try, select_features, target, classifiers, iters, testSize):
    
    # is for evaluating classification models. 
    
    # classifiers takes in a list of possible classifiers and features.
    
    
    
    X = df
    y = df.target
    
    perf_dfs = []
    
    for i in range(0, iters):
        
        features_to_use = base_features + random.sample(features_to_try, select_features)
        X_sub = X[features_to_use]
        
        
        X_train, X_test, y_train, y_test = train_test_split(X_sub, y, test_size=testSize)
        
        
        for clf in classifiers:
            
            model = clf
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            
            
            model_type = []
            accs = []
            feats = [] 
            coefs = []
            
            if 'Logistic' in str(clf):
                for feat, coef in zip(X_train.columns, model.coef_[0]):
                    model_type.append(str(clf))
                    accs.append(accuracy_score(y_test, preds))
                    feats.append(feat)
                    coefs.append(abs(coef))
                    
            if 'RandomForestClassifier' in str(clf):
                for feat, coef in zip(X_train.columns, model.feature_importances_):
                    model_type.append(str(clf))
                    accs.append(accuracy_score(y_test, preds))
                    feats.append(feat)
                    coefs.append(abs(coef))
            
            
            feature_coef_dict = {
            'model_type': model_type,
            'accuracy': accs,
            'feature':feats,
            'coefs': coefs}
            
            perf_dfs.append(pd.DataFrame(feature_coef_dict))
            
    performance_df = pd.concat(perf_dfs)
    
    return performance_df

##################################################################################################################################


# a create important boolean features function
     # it returns a dataframe with extra "dummy" features that combines feature and columns
     # for example is the house big and in a good neighborhood and has a lot of bathrooms


# outlier detection function. can remove outlier rows or can add "is outlier" function. 

# create a function to set display options.
