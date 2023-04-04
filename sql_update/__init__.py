from sql_update.connect_mysql import sql_postman
sql_postman_=sql_postman(host="localhost",user="vivek",password="password",database="mydb",conversion_dict='/home/pooja/PycharmProjects/stock_valuation/codes/sql_update/codes_to_sql_dict.csv')
dc=sql_postman_.sql_dict