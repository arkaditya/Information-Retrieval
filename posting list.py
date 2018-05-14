#Python 3.6
#Author: Arkaditya Verma Date: 05-Feb-2018
import re
import os
import collections
import time

class index:

    def __init__(self, path):
            self.path = path
            self.list_docs = os.listdir(self.path)
            self.buildIndex()

    def buildIndex(self):

        start_time = time.time()
        #function to read documents from collection, tokenize and build the index with tokens
        #index should also contain positional information of the terms in the document --- term: [(ID1,[pos1,pos2,..]), (ID2, [pos1,pos2,…]),….]
        #use unique document IDs

        self.doc_list = {}

        # Multidimensional dictionary
        self.dic = collections.defaultdict(lambda :collections.defaultdict(list))

        for docId,filename in enumerate(self.list_docs):
            f = open(os.path.join(self.path, filename), 'r')
            self.doc_list[filename] = docId
            word_seq = 1
            for line in f:
                #Tokenize each line in the opened file
                line = line.replace(".", "")
                line = line.lower()
                line = re.split(r'\d+|\W+', line)

                for words in line:
                    self.dic[words][docId].append(word_seq)
                    word_seq += 1
            f.close()
        self.dic.pop('')

        end_time = time.time()
        print("Index built in %s seconds" %(end_time - start_time))

    def and_query(self, query_terms):
        #function for identifying relevant docs using the index
        start_time = time.time()
        print("Result for the query: %s" %(' AND '.join(query_terms)))
        #Sort the words in the list and convert them to lower case for better searching
        query_terms = sorted(query_terms,key=len)
        query_terms = [word.lower() for word in query_terms]

        # store each word's docID list in  the query-docs_list and evaluate merge_intersect using the lists in query_doc_list one at a time
        query_docs_list = {word: ([docId for docId in self.dic[word]]) for word in query_terms}
        result_list = query_docs_list[query_terms[0]]
        for words in query_terms[1:]:
            result_list = self.merge_intersect(result_list,query_docs_list[words])

        if len(result_list) == 0:
            print("No such Document found with these set of words ")
        else:
            print("Total Docs retrieved: ", len(result_list))
            for i in result_list:
                x = list(self.doc_list)
                print(x[i])

        end_time = time.time()
        print("Retrieved in %s seconds \n" % (end_time - start_time))

    def print_dict(self):
        # function to print the terms and posting list in the index in sorted order
        for words in sorted(self.dic):
            print("%s  :  %s" %(words,list(self.dic[words].items())))

    def print_doc_list(self):
        # function to print the documents and their document i
        print("     ")
        for docId,filename in enumerate(os.listdir(self.path)):
            print("DocID %s ==> %s" %(docId,filename))
        print("\n")

    def merge_intersect(self,list1,list2):
        intersect = []
        i = 0
        j = 0
        while i < len(list1) and j < len(list2):
            if list1[i] == list2[j]:
                intersect.append(list1[i])
                i += 1
                j += 1
            elif list1[i] < list2[j]:
                i += 1
            else:
                j += 1
        return intersect

if __name__ == '__main__':
    a = index('collection/')
    a.print_dict()
    a.print_doc_list()
    a.and_query(['with','without','yemen'])
    a.and_query(['with','without','yemen','yemeni'])
    a.and_query(['India','Us','anD'])     #Special case if query term is combination of both uppercase and lowercase terms
    a.and_query(['US','UN'])
    a.and_query(['KOLWEZI','OF'])
    a.and_query(['.','" "','|'])         #Search for Special characters if any