import pandas as pd
import graphlab as gl
import matplotlib.pyplot as plt
import numpy as np
from graphlab.toolkits.cross_validation import cross_val_score, KFold

def recommend(ratings, for_prediction, output_fname, num_factors=8):
    rec_engine = gl.factorization_recommender.create
    kfold = KFold(ratings, 5)
    param_dict = {'observation_data':'ratings',
                    'user_id':"user_id",
                    'item_id':"joke_id",
                    'target':'rating',
                    'solver':'auto',
                    'num_factors':num_factors}
    job = cross_val_score(kfold, rec_engine, param_dict)
    score = job.get_results()
    return score
    #print "Score for {} factors = {}".format(num_factors, score)
    #sample_sub.rating = rec_engine.predict(for_prediction)
    #sample_sub.to_csv(output_fname, index=False)

if __name__ == "__main__":
    sample_sub_fname = "../data/sample_submission.csv"
    ratings_data_fname = "../data/ratings.dat"
    output_fname = "../data/test_ratings.csv"

    ratings = gl.SFrame(ratings_data_fname, format='tsv')
    sample_sub = pd.read_csv(sample_sub_fname)
    for_prediction = gl.SFrame(sample_sub)


    '''
    rec_engine.training_rmse #3.045822908400789
    len(ratings['joke_id'].unique()) #141
    len(ratings['user_id'].unique()) #50692
    plt.hist(ratings['rating']) #This doesn't work
    plt.show()
    '''
    param = 8
    score = recommend(ratings, for_prediction, output_fname, param)
