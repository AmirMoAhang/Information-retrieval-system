from collections import defaultdict

from src.preprocessing import preprocessing

def build_positional_indexing(docs):
    index = defaultdict(lambda: defaultdict(list))
    for doc_id, text in docs.items():
        tokens = preprocessing(text)
        for pos, token in enumerate(tokens):
            index[token][doc_id].append(pos)
    return index


def build_permuterm_index(positional_index):
        permuterm_index = {}
        for term in positional_index:
            term_rot = term + "$"
            for i in range(len(term_rot)):
                rotated = term_rot[i:] + term_rot[:i]
                permuterm_index[rotated] = term
        return permuterm_index




# # Test
# if __name__ == '__main__':
#     from load_documents import load_documents
#     docs = load_documents('DOCs/Test')
#     print(build_positional_indexing(docs))