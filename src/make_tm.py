import argparse
import glob
from lda import LDA
import os
import pandas as pd
import pickle
import spacy
from tqdm import tqdm



def load_corpus(newspaper, data_path):
    '''
    function to load cleaned newspaper pickle

    '''
    df = pd.read_pickle(os.path.join(data_path, '{}.pkl').format(newspaper))
    corpus = df['text'].values.tolist()
    dates = df['date'].values
    print('Corpus Loaded!')
    
    return corpus, dates


def calculate_theta(model):
    '''
    function to calculate theta (doc/topic prob matrix)
    '''
    print('calculating theta')
    theta_df = pd.read_csv(model.model.fdoctopics(), delimiter='\t', header=None)
    theta_df.drop(theta_df.iloc[:, :2], inplace=True, axis=1)
    return theta_df.values


def export_model(model, model_path, dates, k, newspaper):
    print("\n[INFO] writing content to file...\n")
    with open(os.path.join(model_path, f"{newspaper}_{k}_content.txt"), "w") as f:
        for topic in model.model.show_topics(num_topics=-1, num_words=10):
            f.write("{}\n\n".format(topic))


    theta = calculate_theta(model)

    print("[INFO] exporting model...")
    out = dict()
    out["model"] = model.model
    out["dictionary"] = model.dictionary
    out["corpus"] = model.corpus
    out["theta"] = theta
    out["dates"] = dates

    with open(os.path.join(model_path, f"{newspaper}_{k}_model.pcl"), "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    
def lemmatize(corpus):
    '''
    Lemmatizer using SpacyNLP
    Be sure to adapt n_processes based on your specs!
    '''

    lemmatized_corpus = []
    print('Lemmatizing.....')
    for doc in tqdm(nlp.pipe(corpus, batch_size=32, n_process=25), total = len(corpus)):
        lemmatized_corpus.append([token.lemma_ for token in doc if token.lemma_ is not None])
    return lemmatized_corpus


def make_tm(newspaper, data_path, model_path, k):
    '''
    Create topic model using mallet. 
    You need to specify the location of your MaLLET installation in `os.environ`

    '''
    corpus, dates = load_corpus(newspaper, data_path)
    os.environ['MALLET_HOME'] = '/work/nl-jump-entropy/event-flow/src/mallet/'
    # check if lemmatized corpus exists, time saver!

    lemmatized_data_path = os.path.join(args.data_path, '../data/lemmatized_data')

    if os.path.isfile(os.path.join(lemmatized_data_path, f'{newspaper}_lemmatized.pcl')):
        with open(os.path.join(lemmatized_data_path, f'{newspaper}_lemmatized.pcl'), "rb") as fobj:
            lemmatized_corpus = pickle.load(fobj)
    else:
        lemmatized_corpus = lemmatize(corpus)
        with open(os.path.join(lemmatized_data_path, f"{newspaper}_lemmatized.pcl"), "wb") as f:
            pickle.dump(lemmatized_corpus, f, protocol=pickle.HIGHEST_PROTOCOL)

    ## Make the topic model using LDA
    lda_model = LDA(lemmatized_corpus)
    ## If K is not given, we will optimize for K (can be slow)
    if k:
        print(f'Default K of {k}')
    else:
        print('Optimizing K')
        k, _ = lda_model.coherence_k(krange=list(range(20, 150, 10)))
        print("[INFO] optimal number of topics: {}".format(k))
    lda_model = LDA(lemmatized_corpus, k=k)
    lda_model.fit()

    export_model(lda_model, model_path, dates, k, newspaper)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--newspaper', type=str)
    parser.add_argument('--data_path', type=str, default='../data/datasets')
    parser.add_argument('--model_path', type=str, default='../models')
    parser.add_argument('--k', type=int) #if empty will optimize K
    args = parser.parse_args()

    lemmatized_data_path = os.path.join(args.data_path, '../data/lemmatized_data')
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(lemmatized_data_path):
        os.makedirs(lemmatized_data_path)

    nlp = spacy.load('nl_core_news_lg', disable=['ner', 'parser', 'tagger'])

    make_tm(args.newspaper, args.data_path, args.model_path, args.k)
