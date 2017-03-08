import pandas as pd
import graphlab as gl
import numpy as np

def load_csv(filename):
    return pd.read_csv(filename)

def load_dat(filename):
    return pd.read_table(filename)

def single_value_decomposition(df):
    pass

if __name__ == "__main__":

    '''
    ---------TRAINING DATA -------------
    '''
    ratings_contents = load_dat("data/ratings.dat")

    '''
    ---------TEST DATA -------------
    '''
    test_data = load_csv("data/sample_submission.csv")
    '''
    --------- Single Value Decomposition -------------
    '''
    U,Sigma,VT = np.linalg.svd(ratings_contents.as_matrix())
