import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

#챗봇을 호출하는 문장이 맞는지 확인(느낌표 check)
# 챗봇과 연동시 필요 없어질 수도 있음(파실할 떄 check가 가능하다면)

def query_confirm(query_input):
    global status_value
    if(query_input[0:1] == "!" or query_input[0:2] == "!!"):
        status_value= True
        return True
    else:
        status_value= False
        return False


#!를 제거 하는 함수
def delete_exclamation_mark(input):
    if(input[0:1] == "!"):
        return input[1:]
    elif(input[0:2] == "!!"):
        return input[2:]


#서버에서 코퍼스 불러오기 (서버 Copus Load)
data = pd.read_csv("test_dummy_first.csv")

#질문내용 추출 (서버 Copus 받는 형태에 따라 변경)
corpus = data['질문내용'].values.tolist()

#답변 내용 추출 (서버 Copus 받는 형태에 따라 변경)
answer_corpus = data['답변내용'].values.tolist()

#문장 입력 받기 (서버에서 받는 Input으로 변경)
query = input("느낌표로 시작하는 문장을 입력해주세요")

#intent 분석 (따로 필요없을 경우 제거)
query = delete_exclamation_mark(query)

# embedding 모듈 load
embedder = SentenceTransformer('xlm-r-bert-base-nli-stsb-mean-tokens')

# query와 copus Embedding
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
query_embedding = embedder.encode(query, convert_to_tensor=True)


# cosine_similarity 계산
cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]

#상위 5개 추출
top_k = 5
top_results = np.argpartition(-cos_scores, range(top_k))

for i, idx in enumerate(top_results[0:top_k]):
    print(str(i+1)+"번 "+corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))

#문장 선택 (챗봇에 해당되는 코드로)
find_answer = input("찾고 싶으신 질문은 어떤것 입니까?(1~5번 입력)")

#답변 (String으로 서버로 넘기기)
print(answer_corpus[top_results[int(find_answer)-1]])