
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans


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


# a feature importance and performance evaluation function
     # will return a dataframe to analyze to find out what features are good at what
     
# a create important boolean features function
     # it returns a dataframe with extra "dummy" features that combines feature and columns
     # for example is the house big and in a good neighborhood and has a lot of bathrooms
     
# a function to split a pandas datetime column into a different date columns (year, month, day, day of week, day of year)

# outlier detection function. can remove outlier rows or can add "is outlier" function. 


        
