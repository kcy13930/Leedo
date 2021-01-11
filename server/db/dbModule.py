import pymysql

class Database():
    def __init__(self):
        self.db = pymysql.connect(host='',
                                  user='',
                                  password='',
                                  db='',
                                  port=,
                                  charset='utf8')

        self.cursor= self.db.cursor(pymysql.cursors.DictCursor)
    def execute(self, query, args={}):
        self.cursor.execute(query, args)

    def execute_one(self, query, args={}):
        self.cursor.execute(query, args)
        row = self.cursor.fetchone()
        return row

    def execute_all(self, query, args={}):
        self.cursor.execute(query, args={})
        row = self.cursor.fetchall()
        return row

    def commit():
        self.db.commit()

db_class = dbModule.Database()
sql = "SELECT QAID, question_context FROM leedo.qa_dataset"
result = db_class.execute_all(sql)
print(result)
