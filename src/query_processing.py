from nltk.stem import PorterStemmer

from src.indexing import build_positional_indexing, build_permuterm_index
from src.load_documents import load_documents
from src.preprocessing import preprocessing

class QueryProcessing:
    
    def __init__(self, path):
        self.ps = PorterStemmer()
        self.docs = load_documents(path)
        self.positional_index = build_positional_indexing(self.docs)
        self.permuterm_index = build_permuterm_index(self.positional_index)
        self.all_docs = self.get_all_docs()
        
    def get_all_docs(self):
        return set(self.docs.keys())
    
    
    def get_docs(self, word):
        word = word.lower()
        word = self.ps.stem(word)
        return set(self.positional_index[word].keys()) if word in self.positional_index else set()
    
    
    def boolean_query(self, query):
        
        all_docs_tmp = self.all_docs.copy()
        
        tokens = preprocessing(query)

        if len(tokens) == 0:
            return set()
        
        elif len(tokens) == 1:
            return self.get_docs(tokens[0])
            
        elif tokens[0].upper() == "NOT":
            result = all_docs_tmp - self.get_docs(tokens[1])
            i = 2
            
        else:
            result = self.get_docs(tokens[0])
            i = 1

        while i < len(tokens):
            op = tokens[i].upper()  
            next_token = tokens[i + 1]

            if op == "AND":
                if i + 2 < len(tokens) and tokens[i + 2].upper() == "NOT":
                    result &= (all_docs_tmp - self.get_docs(tokens[i + 3]))
                    i += 3
                else:
                    result &= self.get_docs(next_token)
                    i += 2
            elif op == "OR":
                if i + 2 < len(tokens) and tokens[i + 2].upper() == "NOT":
                    result |= (all_docs_tmp - self.get_docs(tokens[i + 3]))
                    i += 3
                else:
                    result |= self.get_docs(next_token)
                    i += 2
            elif op == "NOT":
                result -= self.get_docs(next_token)
                i += 2
            else:
                raise ValueError(f"Unknown operator: {op}")

        return result
    
    
    def phrase_query(self, phrase):
        
        tokens = preprocessing(phrase)
        
        if not tokens:
            return set()

        first_word = tokens[0]
        if first_word not in self.positional_index:
            return set()
        candidate_docs = set(self.positional_index[first_word].keys())


        for i in range(1, len(tokens)):
            word = tokens[i]
            if word not in self.positional_index:
                return set()
            docs_with_word = set(self.positional_index[word].keys())
            candidate_docs &= docs_with_word

            valid_docs = set()
            for doc_id in candidate_docs:
                prev_positions = self.positional_index[tokens[i-1]][doc_id]
                curr_positions = self.positional_index[word][doc_id]

                for pos in prev_positions:
                    if (pos + 1) in curr_positions:
                        valid_docs.add(doc_id)
                        break
                
                candidate_docs |= valid_docs

        return candidate_docs
    

    def wildcard_query(self, query):
        
        if "*" not in query:
            return set(self.positional_index[query].keys()) if query in self.positional_index else set()

        parts = query.split("*")
        if len(parts) > 2:
            raise NotImplementedError("فقط یک '*' پشتیبانی می‌شود")

        rotated_query = query.replace('*', '$')
        
        matched_terms = set()
        for rotated, term in self.permuterm_index.items():
            if rotated.startswith(rotated_query):
                matched_terms.add(term)


        results = set()
        for term in matched_terms:
            results.update(self.positional_index[term].keys())

        return results
    
    
    
    
    
    
    
    
    