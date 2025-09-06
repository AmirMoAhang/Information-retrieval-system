from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer 


def preprocessing(text):
    
    # lower case
    text = text.lower()
    
    
    # Tokenizing
    tokens = word_tokenize(text)
    
    
    # delete stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    
    # stemming with porter algorithm
    ps = PorterStemmer()
    stmmed_tokens = [ps.stem(word) for word in tokens]
    
    return stmmed_tokens


# if __name__ == '__main__':
#     print(preprocessing('testing test # number things this'))
