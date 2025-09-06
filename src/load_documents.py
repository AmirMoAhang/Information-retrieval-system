import os

def load_documents(path):
    documents={}
    for indx, filename in enumerate(os.listdir(path)):
        if filename.endswith('.txt'):
            filepath = os.path.join(path, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                documents[filename.split('.')[0]] = file.read()
                
    return documents
            


# if __name__ == '__main__':            
#     print(load_documents('DOCs/Test'))
    
