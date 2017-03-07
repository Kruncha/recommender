import pandas as pd
import graphlab as gl
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sample_sub_fname = "../data/sample_submission.csv"
    ratings_data_fname = "../data/ratings.dat"
    output_fname = "../data/test_ratings.csv"

    ratings = gl.SFrame(ratings_data_fname, format='tsv')
    sample_sub = pd.read_csv(sample_sub_fname)
    for_prediction = gl.SFrame(sample_sub)
    rec_engine = gl.factorization_recommender.create(observation_data=ratings,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     solver='auto',
                                                     regularization=.01)

    sample_sub.rating = rec_engine.predict(for_prediction)
    sample_sub.to_csv(output_fname, index=False)
    '''

    rec_engine.training_rmse #3.045822908400789
    len(ratings['joke_id'].unique()) #141
    len(ratings['user_id'].unique()) #50692
    plt.hist(ratings['rating']) #This doesn't work
    plt.show()
    '''
