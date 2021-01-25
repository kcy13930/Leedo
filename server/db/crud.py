import dbModule as db
import pandas as pd

new_data=pd.read_json('')

# 데이터 로드
database = db.Database()
qa_sql = "SELECT * FROM leedo.qa_dataset"
ig_sql = "SELECT * FROM leedo.imground"
dataset_qa = database.execute_all(qa_sql)
dataset_ig = database.execute_all(ig_sql)
dataset_qa = pd.DataFrame(dataset_qa)
dataset_ig = pd.DataFrame(dataset_ig)


qa_sql_create = "INSERT INTO `qa_dataset` (`QAID`,`question_ts`,`answer_ts`,`question_message_id`,`answer_message_id`,`questioner_id`,`answerer_id`,`questioner_name`,`answerer_name`,`question_context`,`answer_context`,`channel`) VALUES ();"
