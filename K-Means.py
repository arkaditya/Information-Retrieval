# Python 3.0
""" ARKADITYA VERMA
    A20414100"""
import time
import collections
import math
import os
import re
import random
import operator
import numpy as np
from numpy import linalg as LA
##import matplotlib.pyplot as plot

# import other modules as needed
# implement other functions as needed
class kmeans:

    def __init__(self, path_to_collection):
        self.path = path_to_collection
        self.list_docs = os.listdir(self.path)
        self.idf_dic = collections.defaultdict()
        self.doc_name_dict = {}
        self.tfidf_dic = {}
        self.doc_vector = {}
        self.doc_terms = {}
        self.doc_tfidf = {}
        self.documents = []
        self.doc_freq = collections.defaultdict(int)
        self.stop_words = ('a', 'an', 'and', 'are', 'as', 'at',
                           'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is',
                           'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
                           'will', 'with', '')
        self.doc_weight = {}
        self.readDocuments()
        self.createindex = self.buildIndex()
        self.compute_doc_frequency()
        self.doc_len = self.calculate_doc_lengths()
        self.doc_length = self.compute_doc_length()
        self.dic = sorted(self.createindex.keys())
        self.document_vector = self.calculate_doc_vector()


    def readDocuments(self):

        self.doc_list = []
        # Multidimensional dictionary
        self.dic = collections.defaultdict()

        for docId, filename in enumerate(self.list_docs, 1):
            f = open(os.path.join(self.path, filename), 'r')
            word_seq = 0
            lines = f.read()
            lines = self.tokenize(lines)
            term_list = set(lines) - set(self.stop_words)
            self.doc_terms[docId] = term_list
            self.doc_tfidf[docId] = {}
            self.documents.append(lines)
            self.doc_name_dict[docId] = filename
            self.doc_list.append(docId)
            f.close()

    def buildIndex(self):

        index = collections.defaultdict(list)
        for i, tokens in enumerate(self.documents):
            terms = set(tokens)
            for toks in terms:
                index[toks].append([i+1, float(tokens.count(toks))])
        return index

    def calculate_doc_lengths(self):
        doc_length = collections.defaultdict(float)

        for key, list1 in self.createindex.items():
            for docID, lent in list1:
                doc_length[docID] += int(lent)

        return doc_length

    def compute_doc_frequency(self):
        for tokens in self.documents:
            for word in set(tokens):
                self.doc_freq[word] += 1.0

    def compute_doc_length(self):
        N = len(self.documents)
        docNorms = collections.defaultdict(int)
        for word, lists in self.createindex.items():
            for docID, tf in lists:
                df_weight = float(math.log10(N / self.doc_freq[word])) if self.doc_freq[word] > 0 else 0.0
                tf_weight = float(1.0 + math.log10(tf))

                docNorms[docID] = docNorms[docID] + ((tf_weight * df_weight) ** 2)

        return {key: float(value ** 0.5) for key, value in docNorms.items()}

    def calculate_doc_length(self):

        doc_length = collections.defaultdict(float)
        for key, list1 in self.createindex.items():
            for docID, lent in list1:
                doc_length[docID] += float(lent)
        return doc_length

    def calculate_doc_vector(self):
        return [self.query_vector(doc) for doc in self.documents]

    def vector_t(self, vector):
        query_dict = []
        count = collections.Counter(vector)
        for el in self.dic:
            query_dict.append(count[el])

        return query_dict

    def tfidf_index(self):
        N = len(self.list_docs)
        for words, item in self.dic.items():
            idf = math.log10(N / len(item))
            self.idf_dic[words] = idf
            # print(item)
            for k, v in item:
                if words not in self.tfidf_dic.keys():
                    tfidf_weight = idf * (1 + math.log10(len(v)))
                    self.tfidf_dic[words] = [(k, tfidf_weight)]
                else:
                    tfidf_weight = idf * (1 + math.log10(len(v)))
                    temp = (k, tfidf_weight)
                    self.tfidf_dic[words].append(temp)

        for term, tuple_list in self.tfidf_dic.items():
            for doc, value in tuple_list:
                self.doc_tfidf[doc][term] = value
                self.doc_weight[doc] += value**2


    @staticmethod
    def tokenize(list_words):

        line = list_words.replace(".", "")
        line = line.lower()
        line = re.split(r'\d+|\W+', line)
        return line

    def word_list(self, docs):

        word_list = set()
      #  docs = [int(i) for i in docs]
        for docid in docs:
            word_list = word_list.union(set(self.doc_terms[docid]))
        sortedtokens = list(word_list)
        return sortedtokens

    def cosine_score(self, query_vector):
        # calculate IDF
        def idf(term):
            df = float(self.doc_freq[term])
            N = float(len(self.documents))

            try:
                return math.log10(N / df)
            except:
                return 0.0

        scores = collections.defaultdict(int)
        for id, value in enumerate(query_vector):
            each_query = self.dic[id]
            if each_query in self.createindex:
                for lists in self.createindex[each_query]:
                    docID, tf_val = lists
                    tf_weight = (1.0 + math.log10(tf_val)) * idf(each_query)
                    scores[docID] += (tf_weight * value)
        for docID in scores:
            scores[docID] = scores[docID] / self.doc_length[docID] if self.doc_length[docID] > 0 else 0.0
        return scores

    def compute_RSS(self, centroids, clusters):
        avg_rss = 0
        print("Cluster\t\tDocID\t\t\tRSS")
        print("===========================================================")

        # Iterate all centroids
        for i, centroid in enumerate(centroids):
            rss = 0
            cen = np.array(centroid)
            minn = math.inf
            nearest_doc = 0
            for cluster in clusters[i]:
                c = np.array(self.document_vector[cluster])
                # Calculating L2 norm
                norm = LA.norm(np.subtract(cen, c))

                if minn > norm:
                    minn = norm
                    nearest_doc = cluster

                minn = min(minn, norm)
                rss = np.add(rss, norm)
            print("%s\t\t%s\t\t\t%s" %(i,nearest_doc,rss))
            avg_rss += rss
        return avg_rss

    def query_vector(self, query_terms):

        vector = collections.defaultdict(float)
        N = len(self.documents)
        for query in query_terms:
            df = self.doc_freq[query]
            if df > 0:
                vector[query] = math.log10(N / df)
        return self.vector_t(vector)

    def update_centroids(self, clusters, centroid):
        for cluster in clusters.keys():
            c = centroid[cluster]
            for d in clusters[cluster]:
                for i, tok in enumerate(self.document_vector[d]):
                    c[i] += tok

            for i, tok in enumerate(c):
                c[i] /= len(clusters[cluster])

        return centroid

    def vector_t(self, vector):

        query_dict = []
        count = collections.Counter(vector)
        for el in self.dic:
            query_dict.append(count[el])

        return query_dict

    def create_cluster(self, centroids):
            ds = [self.cosine_score(c) for c in centroids]

            d = collections.defaultdict(float)
            for k, _ in enumerate(self.documents):
                n = [d[k] for d in ds]
                d[k] = n.index(min(n))

            out = collections.defaultdict(list)
            for k, v in d.items():
                out[v].append(k)

            return out

    def clustering(self, kvalue):

        if len(self.documents) < kvalue:
            kvalue = len(self.documents)

        centroids = [self.query_vector(c) for c in random.sample(self.documents, kvalue)]

        avg_rss = 0
        for i in range(5):
            clusters = self.create_cluster(centroids)
            centroids = self.update_centroids(clusters, centroids)
            rss = self.compute_RSS(centroids, clusters)
            avg_rss += rss
            print("\n")
        print("RSS value:", avg_rss/5)
        return centroids, avg_rss/5

if __name__ == '__main__':

    start_t = time.clock()

    N = input("Enter the no of clusters : ")
    a = kmeans('kmeans_collection/')

    for i in range(2, int(N)+1):
        start = time.clock()
        print("\nNumber of clusters:", i)
        avg = a.clustering(i)
        print("\nClustering time : ", round(time.clock() - start, 3), "seconds")

    print("################## CLUSTERING FOR k = 2 to 30 ################")

    time_init = time.clock()
    avg_rss = collections.defaultdict(tuple)
    start = time.clock()
    for i in range(2,31):
        start = time.clock()
        print("\nNumber of clusters:", i)
        centroid,avg = a.clustering(i)
        avg_rss[i] = [avg, round(time.clock() - start, 3)]
        print("\nClustering time  for compute: ", round(time.clock() - start, 3), "seconds")

    print("\nTotal Clustering time : ", round(time.clock() - time_init, 3), "seconds")
    print("kvalue\t\tRSS")
    print("=========================")
    for kvalue, items in avg_rss.items():
        print(kvalue,"\t",items)