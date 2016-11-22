from gensim import corpora
from gensim.matutils import corpus2dense
from joblib.memory import Memory
import json
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import os.path

from algo.learnspn import LearnSPN


memory = Memory(cachedir="/tmp/spn", verbose=0, compress=9)



@memory.cache
def learn(denseCorpus):
    return LearnSPN(alpha=0.001, min_instances_slice=100, cluster_prep_method="sqrt", ind_test_method="subsample", sub_sample_rows=2000).fit_structure(denseCorpus)

@memory.cache
def loadData(features=100):  
    lmtzr = WordNetLemmatizer()
    
    with open('/home/molina/Dropbox/Datasets/VisualGenome/objects.txt') as f:
        images = f.readlines()
        
        stopw = set(stopwords.words('english'))
        
        texts = [[lmtzr.lemmatize(word) for word in document.lower().split() if word not in stopw] for document in images]
        
        print("documents", len(texts))
        
        dictionary = corpora.Dictionary(texts)
        
        print("dict before filtering", dictionary)
        
        dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=features)
        dictionary.compactify()
        
        print("dict after filtering", dictionary)
        
        
        corpus = []
        
        for text in texts:
            corpus.append(dictionary.doc2bow(text))
        
        denseCorpus = corpus2dense(corpus, num_terms=len(dictionary.keys()))
        
        print(corpus[0])
        print(denseCorpus[0])
        
        
        return corpus, denseCorpus, dictionary
        
if __name__ == '__main__':
    
    if not os.path.isfile('/home/molina/Dropbox/Datasets/VisualGenome/objects.txt'): 
        
        json_data = open("/home/molina/Dropbox/Datasets/VisualGenome/objects.json").read()
    
        data = json.loads(json_data)
        
        f = open('/home/molina/Dropbox/Datasets/VisualGenome/objects.txt', 'w')
        
        for img in data:
            names = [obj["names"] for obj in img["objects"]]
            
            for name in names:
                f.write("_".join(name).replace(".", ""))
                
                f.write(" ")
            f.write("\n")
            
            
        f.close()

    corpus, denseCorpus, dictionary = loadData()
    words = list(dictionary.values())
    print(words)
    
    spn = learn(denseCorpus)
    
    spn.save_pdf_graph(words, outputfile="vgtopics.pdf")
#    spn = LearnSPN(alpha=0.001, min_slices=min_slices, cluster_prep_method="sqrt", ind_test_method=ind_test_method, row_cluster_method=row_cluster_method).fit_structure(train)
