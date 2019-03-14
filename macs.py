
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


def macs_pca_feature(df, cols, pcs, clusters):

     ‘’’
      takes in a df. columns to reduce dimensions, how many dimensions to reduce to, and how many clusters to make.
     ‘’’

     
     