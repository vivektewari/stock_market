import mysql.connector
import pandas as pd


# mydb=mysql.connector.connect(host="localhost",user="vivek",password="password",database="mydb")
# if mydb: print("connection Succesful")
# else :print("connection unsuccesful")
# mycursor=mydb.cursor()
#mycursor.execute("show databases")
# for db in mycursor:
#     print(db)
#pushing historical eod data

class sql_postman(object):
    def __new__(cls,host="localhost",user="vivek",password="password",database="mydb",conversion_dict=None):

        if not hasattr(cls, 'instance'):
            cls.instance = super(sql_postman, cls).__new__(cls)
            cls.obj_count = 0

        else :
            print("sql_postman _instance already exist so returning the existing one")
        return cls.instance
    def __init__(self,host="localhost",user="vivek",password="password",database="mydb",conversion_dict=None):
        if self.__class__.obj_count !=1:
            self.__class__.obj_count=1
            self.mydb=mysql.connector.connect(host=host,user=user,password=password,database=database)
            if self.mydb:
                print("MYSQL connection Successful")
            else:
                print("MYSQL connection unsuccesful")
            self.mycursor = self.mydb.cursor()

            self.sql_dict=pd.read_csv(conversion_dict).set_index("python_codes").to_dict()['sql_reference']
            c=0
    def read(self,sql):
        """
        Implements table read in for : select var1,var2 from table1 where var1=value and var1=value
        :param sql: [table,]
        :return:
        """
        self.mycursor.execute(sql)
        return self.mycursor.fetchall()
    def write(self,sql):
        self.mycursor.execute(sql)
        self.mydb.commit()
if __name__ == "__main__":
    import unittest
    def test_sql_postman_read():

        c = sql_postman(host="localhost",user="vivek",password="password",database="mydb",conversion_dict='/home/pooja/PycharmProjects/stock_valuation/codes/sql_update/codes_to_sql_dict.csv')
        #c = sql_postman(host="localhost", user="vivek", password="password", database="mydb")
        d=c.read("select * from financials")
        #print(d)
    test_sql_postman_read()
