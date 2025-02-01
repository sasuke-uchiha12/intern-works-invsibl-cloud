from vanna.vannadb import VannaDB_VectorStore
from vanna.google import GoogleGeminiChat
import os

from dotenv import load_dotenv

load_dotenv()



class MyVanna(VannaDB_VectorStore, GoogleGeminiChat):
    def __init__(self, config=None):
        MY_VANNA_MODEL = os.getenv("MY_VANNA_MODEL")
        VANNA_API_KEY = os.getenv("VANNA_API_KEY")
        # MY_VANNA_MODEL = 'sasuke12'
        # VANNA_API_KEY = 'fefc5b889d944cd5931916dcff29951c'
        VannaDB_VectorStore.__init__(
            self, vanna_model=MY_VANNA_MODEL, vanna_api_key=VANNA_API_KEY, config=config
        )
        GoogleGeminiChat.__init__(
            self,
            config={
                "api_key": os.getenv("GEMINI_API_KEY"),
                "model": "gemini-1.5-flash",
            },
        )


vn = MyVanna()

vn.connect_to_mysql(
    host="localhost", dbname="testdb", user="root", password="sasuke12", port=3306
)

# Query the information schema for your database structure
df_information_schema = vn.run_sql(
    "SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'testdb'"
)

# Generate a training plan from the information schema
plan = vn.get_training_plan_generic(df_information_schema)
print(plan)

# Uncomment and run this line to train the model based on the generated plan if you approve it
# vn.train(plan=plan)

# Add training data to help the LLM understand your database

# 1. Add a DDL statement for the `users` table
vn.train(
    ddl="""
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        age INT NOT NULL,
        registration_date DATE DEFAULT CURRENT_DATE
    )
"""
)

# 2. Add a DDL statement for the `orders` table
vn.train(
    ddl="""
    CREATE TABLE IF NOT EXISTS orders (
        order_id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        product_name VARCHAR(100) NOT NULL,
        quantity INT NOT NULL,
        order_date DATE DEFAULT CURRENT_DATE,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
"""
)

# 3. Add documentation about your business terminology
vn.train(
    documentation="In our business, each 'user' represents a customer, and 'orders' are purchases made by these customers. 'product_name' specifies the item bought, and 'quantity' refers to the number of units purchased."
)

# 4. Add an example SQL query for retrieving data
vn.train(
    sql="""
    SELECT 
        users.name AS customer_name, 
        orders.product_name, 
        orders.quantity, 
        orders.order_date 
    FROM orders
    INNER JOIN users ON orders.user_id = users.id
    WHERE users.name = 'Naruto Uzumaki';
"""
)

# Inspect the training data to confirm it is correctly set up
training_data = vn.get_training_data()
print(training_data)

# If there is obsolete or incorrect training data, you can remove it
# Example: Remove training data with a specific ID
# vn.remove_training_data(id='1-ddl')


from vanna.flask import VannaFlaskApp

app = VannaFlaskApp(vn)
app.run()
