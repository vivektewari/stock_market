import mysql.connector
mydb=mysql.connector.connect(host="localhost",user="vivek",password="password",database="mydb")
if mydb: print("connection Succesful")
else :print("connection unsuccesful")
mycursor=mydb.cursor()


class auxilary():
    def __init__(self,connection_obj):
        self.conn=connection_obj
    def get_nearest_date(self,interval,forward):