import gensim
from nltk.corpus import stopwords

stop_words = set(stopwords.words('dutch'))

def remove_stopwords(texts):
    '''
    remove accents and increase max length of words
    Dutch has longer words than English
    '''
    
    return ' '.join(word for word in gensim.utils.simple_preprocess(str(texts), max_len=17) if word not in stop_words)


def digit_perc(x):
    return sum(c.isdigit() for c in str(x)) / len(str(x))

def token_filter(token):
    #return not (token.is_punct | token.is_space | token.is_stop | len(token.text) <= 3)
    return not (token.is_space | len(token.text) < 2 | token.is_stop)

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


