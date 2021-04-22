
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel


class LDA:
    def __init__(self, texts, k=5, mallet_path='~/mallet-2.0.8/bin/mallet'):
        self.texts = texts
        self.mallet = mallet_path
        self.k = k

    def generate_dictionary(self, filtering=True):
        return corpora.Dictionary(self.texts)
    
    def generate_corpus(self, dictionary):
        return [dictionary.doc2bow(text) for text in self.texts]
    
    def fit(self):
        print('making dictionary')
        self.dictionary = self.generate_dictionary()
        print('making corpus')
        self.corpus = self.generate_corpus(self.dictionary)
        print('fitting model')
        self.model = gensim.models.wrappers.LdaMallet(self.mallet,
                                                 corpus=self.corpus, 
                                                 id2word=self.dictionary,
                                                 num_topics=self.k,
                                                 workers=6,
                                                 optimize_interval=10,
                                                 random_seed=41
                                                )
        self.coherencemodel = CoherenceModel(model=self.model, 
                                             texts=self.texts,
                                             dictionary=self.dictionary, 
                                             coherence="c_v")
        
        self.coherence = self.coherencemodel.get_coherence()
        
    def coherence_k(self, krange=[10,20,30,40,50], texts=False):
        k_coherences = list()
        for (i, k) in enumerate(krange):
            print("[INFO] Estimating coherence model for k = {}, iteration {}".format(k, i))
            lda_model = LDA(self.texts, k=k)
            lda_model.fit()
            k_coherences.append(lda_model.coherence)


        k_coherences = np.array(k_coherences, dtype=np.float)
        idx =  k_coherences.argsort()[-len(krange):][::-1]
        k = krange[idx[np.argmax(k_coherences[idx]) & (np.gradient(k_coherences)[idx] >= 0)][0]]
        
        return k, k_coherences