# Python 3.6
# Author: Arkaditya Verma Date: 15-March-2018
import re
import os
import collections
import time
import math
import operator
import random
#import numpy


class index:

    def __init__(self, path):
        self.path = path
        self.list_docs = os.listdir(self.path)
        self.idf_dic = collections.defaultdict()
        self.tfidf_dic = {}
        self.query_dic = {}
        self.doc_length_dic = {}
        self.champion_dic = {}
        self.stop_words = ('a', 'an', 'and', 'are', 'as', 'at',
                           'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is',
                           'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
                           'will', 'with', '')
        self.buildIndex()
        self.tfidf_index()
        self.champion_list_index()

    def buildIndex(self):
        #      start_time = time.time()
        # function to read documents from collection, tokenize and build the index with tokens
        # index should also contain positional information of the terms in the document --- term: [(ID1,[pos1,pos2,..]), (ID2, [pos1,pos2,…]),….]
        # use unique document IDs
        self.doc_list = {}

        # Multidimensional dictionary
        self.dic = collections.defaultdict()

        for docId, filename in enumerate(self.list_docs):
            f = open(os.path.join(self.path, filename), 'r')
            word_seq = 0
            lines = f.read()
            lines = self.tokenize(lines)
            for words in lines:
                word_seq += 1
                if words not in self.stop_words:
                    if words in self.dic.keys():
                        if self.dic[words][-1][0] == docId:
                            self.dic[words][-1][1].append(word_seq)
                        else:
                            temp = (docId, [word_seq])
                            self.dic[words].append(temp)
                    elif words not in self.dic.keys():
                        self.dic[words] = [(docId, [word_seq])]
            f.close()


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

    def doc_length(self,docs):

        for docId in docs:
            x = []
            for key,values in sorted(self.tfidf_dic.items()):
                for i,j in values:
                    if i == docId:
                        x.append(j**2)
            self.doc_length_dic[docId] = sum(x)
        #print(self.doc_length_dic)

    def tokenize(self, list_words):
        line = list_words.replace(".", "")
        line = line.lower()
        line = re.split(r'\d+|\W+', line)
        return line

    def process_query(self, query):

        query = query.lower()
        query = re.findall('\w+', query)
        query_list = []
        for words in query:
            if words not in self.stop_words:
                if words in self.idf_dic.keys():
                    self.query_dic[words] = self.idf_dic[words]
                else:
                    self.query_dic[words] = 0
                query_list.append(words)
        return sorted(query_list)

    def cosine_score(self, query, doc_id):

        score = dict()
        for query_term in query:
            if query_term in self.dic:
                    for doc_id,w in self.tfidf_dic[query_term]:
                        score[doc_id] = w      #Using FastCosine Calcualtion, so ignoring wtd for query

        for doc,val in score.items():
            try:
                score[doc] /= self.doc_length_dic[doc]
            except KeyError:
                score[doc] /= 1
        return score

    #METHOD 1 Part A
    def exact_query(self, query_terms,K):
    # function for exact top K retrieval method 1
    # Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
        start_time = time.time()
        score = dict()
        terms = self.process_query(query_terms)

        docs = []
        for qterms in terms:
            for t,values in self.dic.items():
                if qterms == t:
                    for k, v in values:
                        docs.append(k)
        docs = set(docs)

        self.doc_length(docs)
        for docid in docs:
            score[docid] = self.cosine_score(terms,docid)
        score = sorted(self.cosine_score(terms,docs).items(),key=operator.itemgetter(1),reverse=True)[:K]

        print("The Top %s documents for the EXACT query '%s' is:" %(K,query_terms))
        for docs,v in score:
            print(docs)

        stop_time = time.time()
        print("Retrieved EXACT query terms from index in %.4f seconds\n" % (stop_time - start_time))


    # METHOD 2 Starts Here
    def champion_list_index(self):
        #Function to create Champions posting list for highest tfidf docs terms
        r = 8                               #The No of documents selected from the tfidf posting list of terms
        for key, value in self.tfidf_dic.items():
                for k,v in value:
                    new_val = sorted(value,key=operator.itemgetter(1),reverse=True)[:r]
                    self.champion_dic[key] = new_val
        #print(sorted(self.champion_dic.items()))

    def champion_inexact_query(self,query_terms,K):

        start_time = time.time()
        score = dict()
        terms = self.process_query(query_terms)
        docs = []
        for qterms in terms:
            for t,values in self.champion_dic.items():
                if qterms == t:
                    for k, v in values:
                        docs.append(k)
        docs = set(docs)

        self.doc_length(docs)
        for docid in docs:
            score[docid] = self.cosine_score(terms, docid)
        scores = sorted(self.cosine_score(terms, docs).items(), key=operator.itemgetter(1), reverse=True)[:K]

        stop_time = time.time()
        print("The Top %s documents from the Champion List for the query '%s' is:" % (K, query_terms))

        for docs, v in scores:
            print(docs)
        print("Retrieved CHAMPION_LIST query terms from Champion index in %.4f seconds\n" % (stop_time - start_time))
    #METHOD 2  Ends here

    #METHOD 3 Index Elimination
    def index_elimination(self,query,K):

        start_time = time.time()
        terms = self.process_query(query)
        score = dict()
        idf_list = []
        docs = []

        for qterms in terms:
            for t,values in self.idf_dic.items():
                if qterms == t:
                        lis1 = [qterms,values]
                        idf_list.append(tuple(lis1))
        n = len(idf_list)

        idf_list = sorted(idf_list,key=operator.itemgetter(1),reverse=True)[:n/2 if n%2 == 0 else int(n/2+1)]

        for terms,idfs in idf_list:
            for t, values in sorted(self.dic.items()):
                if terms == t:
                    for k,v in values:
                        docs.append(k)
        docs = set(docs)

        self.doc_length(docs)
        for docid in docs:
            score[docid] = self.cosine_score(terms, docid)
        scores = sorted(self.cosine_score(terms, docs).items(), key=operator.itemgetter(1), reverse=True)[:K]

        stop_time = time.time()
        print("The Top %s documents from the Index Elimination for the query '%s' is:" % (K, query))

        for docs, v in scores:
            print(docs)
        print("Retrieved Index Elimination of query terms from the main index in %.4f seconds\n" % (stop_time -
                                                                                                    start_time))

    #METHOD 3 ends here

    #METHOD 4
    def cluster_pruning(self):

        self.leader_dic = {}
        N = len(self.list_docs)
        doc_ids = {x for x,y in enumerate(self.list_docs)}

        leader_size = int(math.ceil(math.sqrt(N)))
        leader_list = random.sample(doc_ids,leader_size)

        list_followers = set(doc_ids) - set(leader_list)
        follower_size = int(math.ceil((math.sqrt(len(list_followers)))))



    def print_dict(self):
        # function to print the terms and posting list in the index in sorted order
        for words in sorted(self.dic):
            print("%s  :  %s" % (words,self.dic[words]))

    def print_doc_list(self):
        # function to print the documents and their document i
        print("     ")
        for docId, filename in enumerate(os.listdir(self.path)):
            print("DocID %s ==> %s" % (docId, filename))
        print("\n")


if __name__ == '__main__':
    a = index('collection/')
    #a.print_dict()
    a.print_doc_list()
    a.exact_query('US and december without india reconsider',5)
    a.champion_inexact_query('US and december without india reconsider',5)
    a.index_elimination('US and december without india reconsider',5)
    #a.cluster_pruning()
    a.exact_query('Ahead of the Family matter', 4)
    a.champion_inexact_query('Ahead of the Family matter', 4)
    a.index_elimination('Ahead of the Family matter', 4)
    a.exact_query('December not the first month', 3)
    a.champion_inexact_query('December not the first month', 3)
    #a.index_elimination('December not the first month', 3)
    a.exact_query('That is not needless be running', 2)
    a.champion_inexact_query('That is needless not be running', 2)
    a.index_elimination('That is needless not be running', 2)
    a.exact_query('Kolwezi Corruption and Rivalries',8)
    a.champion_inexact_query('Kolwezi Corruption and Rivalries', 8)
    a.index_elimination('Kolwezi Corruption and Rivalries', 8)