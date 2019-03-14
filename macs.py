
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans


def macs_pca_features(df, cols, pcs, clusters, prefix):
     
     #add a function/functionality to 
     #vizualize sufficient pcs

     ‘’’
      takes in a df. columns to reduce   dimensions, how many dimensions to reduce to, and how many clusters to make.
     ‘’’
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


———————-———————-———————-———————-


def macs_featImpClf(X, y, clfs, features):

    # going to take in classifiers and 
    # features and return df


    for clf in clfs:
       model = clf
       model.fit(X[features], y)
       
       for feat, imp in zip(features, model.feature_importances_):
          
       
     