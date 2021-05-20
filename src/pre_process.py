
import pandas as pd
import unidecode
from tqdm import tqdm 
import argparse
import utils
import os
import glob
import re


def load_files(input_path, newspaper):
    return glob.glob(os.path.join(input_path, newspaper, '*.tsv'))


def pre_process(newspaper, pages, input_path):
    files = load_files(input_path, newspaper)
    print('Found {} files'.format(len(files)))
    bigFile = []
    regex_pat = re.compile(r'[^a-zA-Z\s]', flags=re.MULTILINE)

    for f in tqdm(files):
        df = pd.read_csv(f, delimiter='\t', usecols=['date', 'page','ocr'])
        df['ocr'] = df['ocr'].astype(str)
        df = df[~df['ocr'].str.contains('objecttype')]
        df = df[df['page'].astype(int) == pages] 

        # filter out pages with just numbers
        df['perc_digits'] = df['ocr'].apply(lambda x: utils.digit_perc(x))
        df = df[df['perc_digits'] <= 0.5]

        df['ocr'] = df['ocr'].apply(lambda x: unidecode.unidecode(x))
        
        df['ocr'] = df['ocr'].str.replace(regex_pat, '')
        df['ocr'] = df['ocr'].str.findall(r'\w{3,}').str.join(' ')

        # filter based on length

        #df['len'] = df['ocr'].astype(str).str.split().apply(len)
        #df = df[df['len'].between(250, 1000, inclusive=True)]

        # remove stop words
        df['ocr'] = df['ocr'].apply(lambda x: utils.remove_stopwords(x))

        df['ocr'] = df['ocr'].str.lower()
        

        df['date'] = pd.to_datetime(df.date, format='%Y-%m-%d')
        # concatenate pages
        df['text'] = df[['date', 'ocr']].groupby('date')['ocr'].transform(lambda x: ' '.join(x))
        df = df[['date', 'text']].drop_duplicates()
        bigFile.append(df)
        
    else:
        pass

    bigFile = pd.concat(bigFile)
    bigFile = bigFile.sort_values(by='date')
    bigFile.to_pickle(f'../data/datasets/{newspaper}.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--newspaper', type=str)
    parser.add_argument('--pages', type=int, default=1)
    #parser.add_argument('-p', '--pages', action='store', dest='alist',
    #                type=str, nargs='*', default=[1],
    #                help="Examples: -i item1 item2, -i item3")
    parser.add_argument('--raw_data_path', type=str, default='../../../news_nl')
    args = parser.parse_args()

    if not os.path.exists('../data/datasets'):
        os.makedirs('../data/datasets')
    
    pre_process(args.newspaper, args.pages, args.raw_data_path)
