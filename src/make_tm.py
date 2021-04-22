
import argparse
import pandas as pd
import os
from lda import LDA
from tqdm import tqdm
import spacy
import pickle


def load_corpus(newspaper, data_path):
    df = pd.read_pickle(os.path.join(data_path, '{}.pkl').format(newspaper))
    corpus = df['text'].values.tolist()
    dates = df['date'].values
    return corpus, dates

def calculate_theta(model):
    print('calculating theta')
    theta_df = pd.read_csv(model.model.fdoctopics(), delimiter='\t', header=None)
    theta_df.drop(theta_df.iloc[:, :2], inplace=True, axis=1)
    return theta_df.values

def export_model(model, dates, k, newspaper):
    print("\n[INFO] writing content to file...\n")
    with open(os.path.join("../models/", "{}_{}_content.txt".format(newspaper, k)), "w") as f:
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

    with open(os.path.join("../models/", "{}_{}_model.pcl".format(newspaper, k)), "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    
    
def lemmatize(corpus):
    lemmatized_corpus = []
    print('Lemmatizing.....')
    for doc in tqdm(nlp.pipe(corpus, n_process=1), total = len(corpus)):
        lemmatized_corpus.append([token.lemma_ for token in doc if token.lemma_ is not None])
    return lemmatized_corpus

def make_tm(newspaper, data_path, k):
    corpus, dates = load_corpus(newspaper, data_path)
    lemmatized_corpus = lemmatize(corpus)
    lda_model = LDA(lemmatized_corpus)
    if k:
        print(f'Default K of {k}')
    else:
        print('Optimizing K')
        k, _ = lda_model.coherence_k(krange=list(range(20, 100, 5)))
        print("[INFO] optimal number of topics: {}".format(k))
    lda_model = LDA(lemmatized_corpus, k=k)
    lda_model.fit()

    

    export_model(lda_model, dates, k, newspaper)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--newspaper', type=str)
    parser.add_argument('--data_path', type=str, default='../data/datasets')
    parser.add_argument('--k', type=int) #if empty will optimize K
    args = parser.parse_args()

    if not os.path.exists('../models'):
        os.makedirs('../models')

    nlp = spacy.load('nl_core_news_lg', disable=['ner', 'tagger'])

    make_tm(args.newspaper, args.data_path, args.k)