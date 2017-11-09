from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
import pandas as pd

if __name__ == "__main__":
    movie_reviews_data_folder = 'txt_sentoken'
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

    # TASK: Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', LinearSVC(C=1000)),
    ])

    # TASK: Build a grid search to find out whether unigrams or bigrams are more useful.
    # Fit the pipeline on the training set using grid search for the parameters

    parameters = {
        'vect__min_df': [0, 1, 2, 3, 4, 5],
        'vect__max_df': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)]
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
    grid_search.fit(docs_train, y_train)

    # TASK: print the mean and std for each candidate along with the parameter
    # settings for all the candidates explored by grid search.
    n_candidates = len(grid_search.cv_results_['params'])
    max_df = []
    min_df = []
    ngram_range = []
    mean = []
    std = []
    for i in range(n_candidates):
        print(i, 'params - %s; mean - %0.2f; std - %0.2f'
              % (grid_search.cv_results_['params'][i],
                 grid_search.cv_results_['mean_test_score'][i],
                 grid_search.cv_results_['std_test_score'][i]))
        max_df.append(grid_search.cv_results_['params'][i]['vect__max_df'])
        min_df.append(grid_search.cv_results_['params'][i]['vect__min_df'])
        ngram_range.append(str(grid_search.cv_results_['params'][i]['vect__ngram_range']))
        mean.append(grid_search.cv_results_['mean_test_score'][i])
        std.append(grid_search.cv_results_['std_test_score'][i])

    data = {'max_df': max_df, 'min_df': min_df, 'ngram_range': ngram_range, 'mean': mean, 'std': std}
    df = pd.DataFrame(data=data)
    df = df[['max_df', 'min_df', 'ngram_range', 'mean', 'std']]
    df = df.sort_values(by=['mean', 'std'], ascending=[0, 1])
    hdf5 = pd.HDFStore('data.h5')
    hdf5['data'] = df
    hdf5.close()
