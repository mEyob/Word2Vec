import numpy as np
import gensim
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans

class Model(object):
    def __init__(self, name, google=False):
        self.name = name
        self.analogical = []
        self.noun_classification = {}
        if google:
            self.model = gensim.models.Word2Vec.load_word2vec_format(name,binary=True)
        else:
            self.model = gensim.models.word2vec.Word2Vec.load(name)
        if not google:
            self.train_time = self.model.total_train_time
        
    def eval_nearest(self, target_words):
        self.nearest_words = pd.DataFrame(columns=["First","Second","Third","Fourth","Fifth"],index=target_words)
        
        for word in target_words:
            try:
                similar_words = self.model.most_similar(word, topn=5)
            except:
                continue
            for i in range(len(similar_words)):
                self.nearest_words.ix[word,i] = similar_words[i][0]
    
    def eval_analogical(self, file):
        self.accuracy = self.model.wv.accuracy(file)
        for i in range(len(self.accuracy)):
            self.analogical.append(round(100*len(self.accuracy[i]["correct"])/(len(self.accuracy[i]["correct"])+len(self.accuracy[i]["incorrect"])),1))
        label = ['capital-world','currency','city-in-state','family','opposite']
        self.analogical = dict(zip(label, self.analogical))
        
    def conc_noun_kmeans(self,word_list):
        self.word_vec = np.zeros([len(word_list), len(self.model.wv.syn0[0])])
        self.word_cluster = pd.DataFrame(word_list,columns=["Words"])
        for i in range(len(word_list)):
            self.word_vec[i] = self.model[word_list[i]]
        for c in [2,3,6]: 
            self.kmeans_clustering = KMeans(n_clusters = c)
            self.cluster = self.kmeans_clustering.fit_predict(self.word_vec)
            self.word_cluster["Clusters_"+str(c)] = self.cluster
        
    def eval_classification(self, concrete_nouns):
        bi_cluster = [self.word_cluster[self.word_cluster["Clusters_2"] == i]["Words"].tolist() for i in [0,1]]
        tri_cluster = [self.word_cluster[self.word_cluster["Clusters_3"] == i]["Words"].tolist() for i in [0,1,2]]
        hex_cluster = [self.word_cluster[self.word_cluster["Clusters_6"] == i]["Words"].tolist() for i in range(6)]
        
        self.bi_class_error(bi_cluster, concrete_nouns)
        self.tri_class_error(tri_cluster, concrete_nouns)
        self.hex_class_error(hex_cluster, concrete_nouns)
        
    def bi_class_error(self, cluster_of_words, concrete_nouns):
        error = []
        for i in range(len(cluster_of_words)):
            cluster = {'natural': 0, 'artifact': 0}
            tmp = concrete_nouns.set_index("Words")
        
            for word in cluster_of_words[i]:
                if tmp.loc[word, "fClusters_2"] == "natural":
                    cluster['natural'] += 1
                elif tmp.loc[word, "fClusters_2"] == "artifact":
                    cluster['artifact'] += 1
            
            count = [ item for item in list(cluster.values()) if item != 0 ]

            if len(count) == 1:
                error.append(0.0)
                continue
            elif len(count) > 1:
                count = sorted(count, reverse=True)
                count.pop(0)
                for val in count:
                    error.append(val)
            
        self.noun_classification["two_class_classification"] = 100 - round(100*sum(error)/concrete_nouns.Words.size, 1)
        
    def tri_class_error(self, cluster_of_words, concrete_nouns):
        error = []
        for i in range(len(cluster_of_words)):
            cluster = {'animal': 0, 'vegetable': 0, 'artifact': 0}
            tmp = concrete_nouns.set_index("Words")
        
            for word in cluster_of_words[i]:
                if tmp.loc[word, "fClusters_3"] == "animal":
                    cluster['animal'] += 1
                elif tmp.loc[word, "fClusters_3"] == "vegetable":
                    cluster['vegetable'] += 1
                elif tmp.loc[word, "fClusters_3"] == "artifact":
                    cluster['artifact'] += 1
            count = [ item for item in list(cluster.values()) if item != 0 ]
        
            if len(count) == 1:
                error.append(0.0)
                continue
            elif len(count) > 1:
                count = sorted(count, reverse=True)
                count.pop(0)
                for val in count:
                    error.append(val)
        self.noun_classification["three_class_classification"] = 100 - round(100*sum(error)/concrete_nouns.Words.size, 1)
        
    def hex_class_error(self, cluster_of_words, concrete_nouns):
        error = []
        for i in range(len(cluster_of_words)):
            cluster = {'bird': 0,'groundAnimal': 0, 'fruitTree': 0, 'green': 0, 'tool': 0, 'vehicle': 0}
            tmp = concrete_nouns.set_index("Words")
        
            for word in cluster_of_words[i]:
                if tmp.loc[word, "fClusters_6"] == "bird":
                    cluster['bird'] += 1
                elif tmp.loc[word, "fClusters_6"] == "groundAnimal":
                    cluster['groundAnimal'] += 1
                elif tmp.loc[word, "fClusters_6"] == "fruitTree":
                    cluster['fruitTree'] += 1
                elif tmp.loc[word, "fClusters_6"] == "green":
                    cluster['green'] += 1
                elif tmp.loc[word, "fClusters_6"] == "tool":
                    cluster['tool'] += 1
                elif tmp.loc[word, "fClusters_6"] == "vehicle":
                    cluster['vehicle'] += 1
            count = [ item for item in list(cluster.values()) if item != 0 ]
        
            if len(count) == 1:
                error.append(0.0)
                continue
            elif len(count) > 1:
                count = sorted(count, reverse=True)
                count.pop(0)
                for val in count:
                    error.append(val)
            
        self.noun_classification["six_class_classification"] = 100 - round(100*sum(error)/concrete_nouns.Words.size, 1)
        
    

def read_conc_nouns():
    cluster = pd.read_csv("Data/eval/ESSLLI2008_concNouns.categorization.dataset_en.txt", sep="\t",usecols=["NOUN","CLASS"])
    tmp = cluster.CLASS.apply(lambda x: x.split("-"))
    df = []
    for item in tmp:
        df.append(dict(zip("fClusters_6 fClusters_3 fClusters_2".split(),item)))
    df = pd.DataFrame(df)

    cluster.drop("CLASS",axis=1,inplace=True)
    cluster.rename(columns={"NOUN":"Words"},inplace=True)
    cluster = cluster.join(df)
    
    return cluster

def load_and_evaluate(modelName, google=False): 
    model = Model(modelName, google)
    
    # Evaluate nearest words
    print("Starting nearest neighbor evaluation task...\n")
    nearest_file = "Data/eval/nearest_neighbor_wordlist.csv"
    eval_nearest = pd.read_csv(nearest_file,header=None,usecols=[1])
    target_words = eval_nearest.loc[:,1].values.tolist()
    
    model.eval_nearest(target_words)
    
    # Evaluate analogical reasoning task
    print("Starting analogical reasoning evaluation task...\n")
    model.eval_analogical("Data/eval/analogical_reasoning_questions-words-short.txt")
    
    
    # Evaluate concrete noun classification task
    print("Starting concrete noun categorization task...\n")
    concrete_nouns = read_conc_nouns()
    word_list = concrete_nouns.Words.tolist()
    model.conc_noun_kmeans(word_list)
    model.eval_classification(concrete_nouns)
    
    return model
       
