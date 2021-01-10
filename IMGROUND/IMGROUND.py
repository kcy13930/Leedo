import pandas as pd
import openpyxl
import numpy as np
from sentence_transformers import SentenceTransformer, util


# 데이터 로드
Dataset=pd.read_json('Leedo_Dataset.json')


# Corpus and Name 셋 세팅
corpus = Dataset['content'].values.tolist()
corpus = [word.replace('\xa0',' ') for word in corpus]

name = Dataset['name'].values.tolist()
name = [word.replace('\xa0',' ') for word in name]

# SentenceTransformer 로드
embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# Input
query = input("느낌표로 시작하는 문장을 입력해주세요")

# 임베딩
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
query_embedding = embedder.encode(query, convert_to_tensor=True)

#코사인 유사도
cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]


#상위 다섯개 출력 (10개 여분으로 세팅하고, 중복 제거)

top_k = 10
top_results = np.argpartition(-cos_scores, range(top_k))

name_overlap=[]

number=1

for i, idx in enumerate(top_results[0:top_k]):
    if number>5:
        break
    if cos_scores[idx] > 0.5:
        if not name[idx] in name_overlap:
            print(str(i+1)+"번 "+corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
            print(str(i+1)+"번 "+name[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
            name_overlap.append(name[idx].strip())
            number +=1

