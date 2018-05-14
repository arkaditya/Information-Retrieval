#Python 3.6
# Author: Arkaditya Verma Date: 05-APRIL-2018
from collections import defaultdict
import re
import os
import collections
import time
import math
import operator
import random

class index:

        def __init__(self, path):
            self.path = path
            self.list_docs = os.listdir(self.path)
            self.idf_dic = collections.defaultdict()
            self.doc_name_dict = {}
            self.tfidf_dic = {}
            self.query_dic = {}
            self.doc_length_dic = {}
            self.doc_terms = {}
            self.doc_tfidf = {}
            self.doc_vector = {}
            self.query_tfidf_vector = {}
            self.stop_words = ('a', 'an', 'and', 'are', 'as', 'at',
                               'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is',
                               'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
                               'will', 'with', '')
            self.relevant = []
            self.nonrelevant = []
            self.sortedtokens = []
            self.buildIndex()
            self.tfidf_index()

        def buildIndex(self):
            #      start_time = time.time()
            # function to read documents from collection, tokenize and build the index with tokens
            # index should also contain positional information of the terms in the document --- term: [(ID1,[pos1,pos2,..]), (ID2, [pos1,pos2,…]),….]
            # use unique document IDs
            self.doc_list = {}

            # Multidimensional dictionary
            self.dic = collections.defaultdict()

            for docId, filename in enumerate(self.list_docs,1):
                f = open(os.path.join(self.path, filename), 'r')
                word_seq = 0
                lines = f.read()
                lines = self.tokenize(lines)
                term_list = set(lines) - set(self.stop_words)
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
                self.doc_terms[docId] = term_list
                self.doc_tfidf[docId] = {}
                self.doc_name_dict[docId] = filename
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

            for term,tuple_list in self.tfidf_dic.items():
                for doc,value in tuple_list:
                    self.doc_tfidf[doc][term] = value


        def doc_length(self, docs):

            for docId in docs:
                x = []
                for key, values in sorted(self.tfidf_dic.items()):
                    for i, j in values:
                        if i == docId:
                            x.append(j ** 2)
                self.doc_length_dic[docId] = sum(x)
            # print(self.doc_length_dic)

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

        def print_dict(self):
            # function to print the terms and posting list in the index in sorted order
            for words in sorted(self.dic):
                print("%s  :  %s" % (words, self.dic[words]))

        def cosine_score(self, query_vec,query_len,docid):

            score = dict()
            #print(doc_list)
            for term in query_vec.keys():
                if term in self.dic:
                    for doc_id, w in self.tfidf_dic[term]:
                        if doc_id not in score.keys():
                            score[doc_id] = w*query_vec[term]    #*query_vector[term]  # Using FastCosine Calcualtion, so ignoring wtd for query
                        else:
                            score[doc_id] += w*query_vec[term]


            for doc, val in score.items():
                den = (query_len * self.doc_length_dic[doc])
                if (den == 0):
                    score[doc] = 0
                else:
                    score[doc] /= den
            return score

        def query_tfidf(self, query):

            #query = self.process_query(query)
            query_len = 0.0
            query_tfidf_vector = {}
            for term in query:
                if term in self.dic.keys():
                    w = 1 + math.log10(query.count(term))
                    query_tfidf_vector[term] = w * self.idf_dic[term]

                else:
                    query_tfidf_vector[term] = 0
            return query_tfidf_vector

        # METHOD 1 Part A
        def exact_query(self, query_terms, K):
            # function for exact top K retrieval method 1
            # Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
            start_time = time.time()
            score = dict()
            terms = self.process_query(query_terms)
            query_vec = self.query_tfidf(terms)               #Generate tfidf for the query

            docs = []
            for qterms in terms:
                for t, values in self.dic.items():
                    if qterms == t:
                        for k, v in values:
                            docs.append(k)
            docs = set(docs)

            self.doc_length(docs)
            for docid in docs:
                score[docid] = self.cosine_score(query_vec,1,docid)

            #query, query_length = self.query_tfidf_vector(query_terms)
            score = sorted(self.cosine_score(query_vec,1, docs).items(), key=operator.itemgetter(1), reverse=True)[:K]

            print("The Top %s documents for the EXACT query '%s' is:" % (K, query_terms))
            for docs, v in score:
                print(docs,self.doc_name_dict[docs])
                #self.exact_doc_list.append(docs)
            stop_time = time.time()

            print("Retrieved EXACT query terms from index in %.4f seconds\n" % (stop_time - start_time))
            #print(self.query_tfidf_vector)

        def doc_term_freq(self):

            for words,items in self.dic.items():
                if words not in self.stop_words:
                    for k,v in items:
                        self.doc_term_freq_dic[words][k] = len(v)
            #print(sorted(self.doc_term_freq_dic.items()))


        def word_list(self,docs):

            word_list = set()
            for docid in docs:
                word_list = word_list.union(set(self.doc_terms[docid]))
            sortedtokens = list(word_list)
            return sortedtokens

        def document_vector(self,docs,sortedtokens):

            doc_vector = {}

            for docid in docs:
                doc_vector[docid] = {}  #Initialize to NULL array

            for docid in docs:
                for terms in sortedtokens:
                    if terms in self.doc_tfidf[docid]:
                        doc_vector[docid][terms] = self.doc_tfidf[docid][terms]
                    else:
                        doc_vector[docid][terms] = 0
                #print(docid,self.doc_vector[docid])
            return doc_vector

        def query_vector(self,query,sortedtokens):

            query_vectorlist = {}
            for terms in sortedtokens:
                if terms in self.dic.keys():
                    query_vectorlist[terms] = (query.count(terms))
                else:
                    query_vectorlist[terms] = 0

            return query_vectorlist

        def rocchio_tfidf(self, query):

            query_len = 0.0
            rocchio_tfidf_vector = {}

            for term in query:
                if term in self.dic.keys():
                    w = 1 + math.log10(query.count(term))
                    rocchio_tfidf_vector[term] = w * self.idf_dic[term]
                    query_len += math.pow(rocchio_tfidf_vector[term], 2)

                else:
                    rocchio_tfidf_vector[term] = 0
            return rocchio_tfidf_vector,query_len

        def rocchio(self, query_terms, pos_feedback, neg_feedback, alpha, beta, gamma):
            # function to implement rocchio algorithm
            # pos_feedback - documents deemed to be relevant by the user
            # neg_feedback - documents deemed to be non-relevant by the user
            # Return the new query  terms and their weights

            relevant_doc_vector = {}
            nonrelevant_doc_vector = {}

            docs_1 = pos_feedback.split()
            relevant = [int(i) for i in docs_1]
            docs_2  = neg_feedback.split()
            non_relevant = [int(i) for i in docs_2]
            words = self.word_list(relevant)

            len_relevant = len(relevant)                               #no. of docs in relevant and non relevant fields
            len_nonrelevant = len(non_relevant)

            q_vector = self.query_vector(query_terms,words)
            relevant_doc_vector = self.document_vector(relevant,words)
            nonrelevant_doc_vector = self.document_vector(non_relevant,words)

            rel_final = {}
            nonrel_final = {}
            opt_query_vector = {}

            for words in q_vector.keys():
                rel_final[words] = 0.0
                nonrel_final[words] = 0.0
                opt_query_vector[words] = 0.0

            for docs,v in relevant_doc_vector.items():
                 for words,idfs in v.items():
                        rel_final[words] = rel_final[words] + relevant_doc_vector[docs][words]

            for docs,v in nonrelevant_doc_vector.items():
                 for words,idfs in v.items():
                        nonrel_final[words] = nonrel_final[words] + nonrelevant_doc_vector[docs][words]

            for words in q_vector.keys():
                opt_query_vector[words] = alpha*q_vector[words] + (beta*rel_final[words])/len_relevant - (gamma*nonrel_final[words])/len_nonrelevant

            #print(opt_query_vector)
            return opt_query_vector

        def rocchio_retreival(self, query_terms, K):

            #start_time = time.time()
            print("+++++++++++++++++++++ ROCCHIO RETRIEVAL ++++++++++++++++++++++++++++++++++++")
            score = dict()
            q_vector,q_len = self.rocchio_tfidf(query_terms)
            print(q_vector)

            docs = []
            for qterms in query_terms:
                for t, values in self.dic.items():
                    if qterms == t:
                        for k, v in values:
                            docs.append(k)
            docs = set(docs)

            self.doc_length(docs)
            for docid in docs:
                score[docid] = self.cosine_score(q_vector,q_len,docid)

            score = sorted(self.cosine_score(q_vector,q_len,docs).items(), key=operator.itemgetter(1), reverse=True)[:K]

            print("\nThe Top %s documents for the ROCCHIO Retreival is:" % (K))
            for docs, v in score:
                print(docs, self.doc_name_dict[docs])


if __name__ == '__main__':

      a = index('collection/')
      #Edit the following query to run for each different query
      query = "RESULTS OF THE POLITICAL POLLS IN BRITAIN REGARDING WHICH PARTY IS IN THE LEAD, THE LABOR PARTY OR THE CONSERVATIVES ."
      a.exact_query(query, 10)
      query = a.process_query(query)
      iteration = 1
      user_input = 'Y'

      while(user_input == 'Y'):

            rel_documents = input("Enter the documents which are relevant: ")
            nonrel_documents = input("Enter the documents which are non relevant: ")
            print("Iteration: %s" % (iteration))
            new_query_list = list(a.rocchio(query, rel_documents, nonrel_documents, 1, 0.75, 0.15).keys())
            a.rocchio_retreival(new_query_list,10)
            print("\n")
            user_input = input("Continue (Y/N): ")
            iteration += 1
