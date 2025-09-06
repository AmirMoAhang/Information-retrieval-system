import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.query_processing import QueryProcessing
from src.ranking import RankingDocuments


import argparse

def run_b(query):
    qp = QueryProcessing('./DOCs')
    print(qp.boolean_query(query))


def run_p(query):
    qp = QueryProcessing('./DOCs')
    print(qp.phrase_query(query))


def run_w(query):
    qp = QueryProcessing('./DOCs')
    print(qp.wildcard_query(query))


def run_g(query, top_k):
    rd = RankingDocuments('./DOCs')
    top_docs = rd.ranking_query(query, top_k)
    print('doc : similarity')
    print('----------------')
    for doc, similarity in top_docs:
        print(f'{doc} : {similarity}')
    
    

def main():
    
    parser = argparse.ArgumentParser(
        description="CLI for IR system"
    )
    
    parser.add_argument("-b", type=str, help="Process binary query")
    parser.add_argument("-p", type=str, help="Process phrase query")
    parser.add_argument("-w", type=str, help="Process wildcard query")
    parser.add_argument("-g", type=str, help="Process Ranking query")
    parser.add_argument("-k", type=int, default=5, help="Top K documents (default=5)")

    args = parser.parse_args()

    if args.b:
        run_b(args.b)

    if args.p:
        run_p(args.p)

    if args.w:
        run_w(args.w)

    if args.g:
        run_g(args.g, args.k)

if __name__ == "__main__":
    main()