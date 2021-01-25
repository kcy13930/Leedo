import pandas as pd
import numpy as np
import os
import re

path = './data/raw_data'
path_all = os.path.join(path,os.listdir(path)[-1])
print(path_all)

def name_matching(qa_df, name_df):
    qa_df['name'] = [name_df[name_df.userid==x].fullname.squeeze() for x in qa_df.user]
    return qa_df

def export_questions(qa_df):
    question_list = qa_df[qa_df['parent_user_id'].isna() & qa_df['replies'].notna()]
    feature=['client_msg_id','text','user_profile','user','name','ts','replies','files']
    question_list = question_list[feature]
    question_list.columns.values[0] = 'question_id'
    question_list
    return question_list

def export_answers(qa_df):
    answer_list = qa_df[qa_df['parent_user_id'].notna()]
    feature=['client_msg_id','text','user_profile','user','name','parent_user_id','ts','files']
    answer_list = answer_list[feature]
    answer_list.columns.values[0] = 'answer_id'
    #answer_list.columns.values[1:] = [x+'_' for x in answer_list.columns.values[1:]]
    return answer_list

# 질문과 답변 join해야함. question.replies의 user, ts가 answer.parent_user_id와 question.user 매칭
def construct_dataset(question_list, answer_list, channel='질문답변'):
    dataset = list()
    for q_index in question_list.index:
        question = question_list.loc[q_index,:]
        join_key = pd.DataFrame(question.replies)
        for jk_user,jk_ts in zip(join_key.user, join_key.ts.astype('float64')):
            answer = answer_list[(answer_list.user==jk_user) & (answer_list.ts==jk_ts)].squeeze()
            if answer.parent_user_id == question.user:
                data = [question['ts'],answer['ts'],
                        question['question_id'],answer['answer_id'],
                        question['user'],answer['user'],
                        question['name'],answer['name'],
                        question['text'],answer['text'], question['files'], answer['files']
                       ]
                dataset.append(data)
    columns = ['질문 날짜','답변 날짜',
               '질문 메세지 id','답변 메세지 id',
               '질문자 id', '답변자 id',
               '질문자 이름', '답변자 이름',
               '질문내용', '답변내용','질문자 file', '답변자 file']
    df = pd.DataFrame(dataset,columns=columns)
    df['채널']=channel
    return df

def export_url(text):
    pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = list(map((lambda x: re.findall(pattern=pattern, string=x)[:-1]), text))
    return text

def clean_str(text):
    pattern = '<[^>]*>'         # HTML 태그 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '\n\n|\n$|^\n'         # 적당히 줄바꿈 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    return text

def export_title(attachments):
    questions = list()
    for attc in attachments:
        if not (type(attc)==np.float):
            attc_df = pd.DataFrame([attc])
            if 'footer' in attc_df.columns:
                questions.append('')
                #question_list.append([pd.DataFrame(attc).footer.squeeze()[1:-1]])
            elif 'title' in attc_df.columns:
                questions.append(attc_df.title.squeeze())
        else:
            questions.append('')
    #return questions
    return questions[0] if len(questions)==1 else ','.join(questions)

def question_init(df):
    df.text = list(map(lambda x: clean_str(x), df.text))
    df['question']=''
    return df

def export_question_of_it_share(df,condition):
    for i in df.index:
        text = df.loc[i,:].text
        if condition=='text': # URL 지우고 TEXT가 남아있으면 그 내용을 질문에 넣음
            df.question[i]= text
        if condition=='section' and not type(df.loc[i,:].attachments)==np.float: # URL 지우고 Text가 비었으면 section에서 내용 가져옴
            title = export_title(df.loc[i,:].attachments)
            df.question[i]=title
    return df

def construct_dataset_it_share(df, channel='IT관련공유'):
    dataset = list()
    for index in df.index:
        sample = df.loc[index,:]
        if len(sample.url)!=0 and sample.question!='':
            data = [sample['ts'],sample['ts'],
                    sample['client_msg_id'],sample['client_msg_id'],
                    sample['user'],sample['user'],
                    sample['name'],sample['name'],
                    sample['question'],sample['url'], sample['files']
                   ]
            dataset.append(data)
    columns = ['질문 날짜','답변 날짜',
               '질문 메세지 id','답변 메세지 id',
               '질문자 id', '답변자 id',
               '질문자 이름', '답변자 이름',
               '질문내용', '답변내용','질문자 file']
    df = pd.DataFrame(dataset,columns=columns)
    df['채널']=channel
    return df


def build_qa_df(path_qa, path_qa_list):
    qa_df = pd.DataFrame()
    for f in path_qa_list:
        f_path = os.path.join(path_qa,f)
        qa_sample_df = pd.read_json(f_path, encoding='utf-8')
        qa_df = pd.concat([qa_df,qa_sample_df], axis=0)
    name_df = pd.read_csv(os.path.join(path,'slack-econovation-2018-members.csv'),encoding='utf-8')
    qa_df = name_matching(qa_df,name_df).reset_index(drop=True)
    question_list = export_questions(qa_df)
    answer_list = export_answers(qa_df)
    qa_df = construct_dataset(question_list, answer_list)
    return qa_df

def build_url_df(path_url, path_list, channel='IT관련공유'):
    df = pd.DataFrame()
    for f in path_list:
        f_path = os.path.join(path_url, f)
        sample_df = pd.read_json(f_path, encoding='utf-8')
        df = pd.concat([df,sample_df],axis=0)

    name_df = pd.read_csv(os.path.join(path,'slack-econovation-2018-members.csv'),encoding='utf-8')
    df = name_matching(df,name_df).reset_index(drop=True)
    df['url'] = export_url(df.text)
    df = question_init(df)
    df_text = export_question_of_it_share(df, condition='text')
    df_text = construct_dataset_it_share(df_text, channel)
    df_section = export_question_of_it_share(df, condition='section')
    df_section = construct_dataset_it_share(df_section, channel)
    df = pd.concat([df_text, df_section],axis=0)
    return df

path_qa = os.path.join(path_all,'공통채널-질문과답변')
path_qa_list = os.listdir(path_qa)

path_it_share = os.path.join(path_all,'공통채널-it관련공유')
path_it_share_list = os.listdir(path_it_share)

path_talk_hard = os.path.join(path_all,'공통채널-기술talk-hard')
path_talk_hard_list = os.listdir(path_talk_hard)

qa_df = build_qa_df(path_qa, path_qa_list)
it_share_df = build_url_df(path_it_share, path_it_share_list, 'IT관련공유')
talk_hard_df = build_url_df(path_talk_hard, path_talk_hard_list, '기술talk-hard')

'''
df = pd.concat([qa_df, it_share_df, talk_hard_df],axis=0).reset_index(drop=True)
df.columns = ['question_ts', 'answer_ts',
              'question_message_id', 'answer_message_id',
              'questioner_id', 'answerer_id',
              'questioner_name', 'answerer_name',
              'question_context', 'answer_context', 'channel'
             ]
df.to_json('qa_dataset.json',orient='records')
'''

df = pd.concat([qa_df, it_share_df, talk_hard_df],axis=0).reset_index(drop=True)
df.to_json('./data/pre_data/dataset.json')
