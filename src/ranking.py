import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.load_documents import load_documents
from src.indexing import build_positional_indexing
from src.preprocessing import preprocessing

class RankingDocuments:
    
    def __init__(self, path):
        self.docs = load_documents(path)
        self.docs_num = len(self.docs)
        self.positional_index = build_positional_indexing(self.docs)
        self.dfs = self.compute_df()
        self.idfs = self.compute_idf()
        self.doc_tf_idf = self.compute_docs_tf_idf()
    
    def compute_df(self):
        df_dict = {}
        for term, postings in self.positional_index.items():
            df_dict[term] = len(postings)
        return df_dict
    
    # smooth idf
    def compute_idf(self):
        idf_dict = {term: (math.log10(self.docs_num / df) + 1) for term, df in self.dfs.items()}
        return idf_dict
    
    
    def compute_docs_tf_idf(self):
        docs = list(self.docs.keys())
        terms = list(self.positional_index.keys())
        tfidf_matrix = {doc_id: {term: 0.0 for term in terms} for doc_id in docs}
        
        for term, postings in self.positional_index.items():
            idf = self.idfs[term]
            
            for doc_id, positions in postings.items():
                tf = len(positions)
                tf_weight = 1 + math.log10(tf) if tf > 0 else 0
                tfidf_matrix[doc_id][term] = tf_weight * idf
        
        return tfidf_matrix
    
    
    def compute_query_tf_idf(self, query):
        terms = list(self.positional_index.keys())
        tf_matrix = {term : 0.0 for term in terms}
        tfidf_matrix = {term:0.0 for term in terms}
        
        tokens = preprocessing(query)
        tokens_count = dict(Counter(tokens))
        
        for term, count in tokens_count.items():
            if term in self.positional_index:
                tf_matrix[term] = (1 + math.log10(count)) if count > 0 else 0
                
        for term in tokens_count.keys():            
            tfidf_matrix[term] = tf_matrix[term] * self.idfs[term]
        
        return tfidf_matrix
    
    
    def compute_cosine_similarity(self, query):
        query_tf_idf = self.compute_query_tf_idf(query)
        query_vec = np.array([[query_tf_idf[term] for term in sorted(query_tf_idf)]])
        
        cosine_similarity_rate = {}
        
        
        for doc_name in self.docs.keys():
            doc_vec = np.array([[self.doc_tf_idf[doc_name][term] for term in sorted(self.doc_tf_idf[doc_name])]])
            cos_sim = cosine_similarity(query_vec, doc_vec)
            cosine_similarity_rate[doc_name] = cos_sim
        
        return cosine_similarity_rate
    
    
    
    def ranking_query(self, query,top_k):
        similarity = self.compute_cosine_similarity(query)
        
        sorted_items = sorted([(name, val[0,0]) for name, val in similarity.items()], key=lambda x: x[1],reverse=True)
        
        
        return sorted_items[:top_k]
    
    
    
    def Test(self):
        print(self.ranking_query('data data diagnosing diseases, detecting malware and optimizing transportation routes', 5))
        # print(self.compute_cosine_similarity('data diagnosing diseases, detecting malware and optimizing transportation routes'))
        # print(self.compute_query_tf_idf('data data diagnosing diseases, detecting malware and optimizing transportation routes'))
        # print(self.compute_tf_idf())
        # print(self.docs.keys())
    
    
    
    
if __name__ == '__main__':
    rd = RankingDocuments('DOCs/Test')
    rd.Test()