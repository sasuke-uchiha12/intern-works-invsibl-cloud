from vanna.openai import OpenAI_Chat
from vanna.vannadb import VannaDB_VectorStore
import os

from dotenv import load_dotenv

load_dotenv()


class MyVanna(VannaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        MY_VANNA_MODEL = os.getenv("MY_VANNA_MODEL")
        VANNA_API_KEY = os.getenv("VANNA_API_KEY")

        VannaDB_VectorStore.__init__(
            self,
            vanna_model=MY_VANNA_MODEL,
            vanna_api_key=VANNA_API_KEY,
            config=config,
        )
        OpenAI_Chat.__init__(self, config=config)


vn = MyVanna(
    config={
        "api_key": os.getenv("OPEN_API_KEY"),
        "model": "gpt-4",
    }
)

vn.connect_to_mysql(
    host="localhost",
    dbname="testdb",
    user="root",
    password="sasuke12",
    port=3306,
)


""" training and only once it should be done!"""

# # # Query the information schema for your database structure
# df_information_schema = vn.run_sql(
#     "SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'testdb'"
# )

# # # # Generate a training plan from the information schema
# plan = vn.get_training_plan_generic(df_information_schema)
# print(plan)

# """---------------------------------------------------------"""

# # Uncomment and run this line to train the model based on the generated plan if you approve it
# # vn.train(plan=plan)

# # Add training data to help the LLM understand your database

# # # 1. Add a DDL statement for the `users` table
# vn.train(
#     ddl="""
#     CREATE TABLE IF NOT EXISTS users (
#         id INT AUTO_INCREMENT PRIMARY KEY,
#         name VARCHAR(100) NOT NULL,
#         email VARCHAR(100) UNIQUE NOT NULL,
#         age INT NOT NULL,
#         registration_date DATE DEFAULT CURRENT_DATE
#     )
# """
# )

# # # # 2. Add a DDL statement for the `orders` table
# vn.train(
#     ddl="""
#     CREATE TABLE IF NOT EXISTS orders (
#         order_id INT AUTO_INCREMENT PRIMARY KEY,
#         user_id INT NOT NULL,
#         product_name VARCHAR(100) NOT NULL,
#         quantity INT NOT NULL,
#         order_date DATE DEFAULT CURRENT_DATE,
#         FOREIGN KEY (user_id) REFERENCES users(id)
#     )
# """
# )

# # # 3. Add documentation about your business terminology
# vn.train(
#     documentation="""
# Our database, `testdb`, consists of two main tables:

# 1. `users`: This table represents customers in our system. It contains:
#    - `id`: A unique identifier for each customer.
#    - `name`: The customer's full name.
#    - `email`: The customer's unique email address.
#    - `age`: The customer's age.
#    - `registration_date`: The date the customer registered, defaulting to the current date.

# 2. `orders`: This table tracks purchases made by customers. It contains:
#    - `order_id`: A unique identifier for each order.
#    - `user_id`: A foreign key referencing the `id` column in the `users` table, linking each order to a customer.
#    - `product_name`: The name of the product purchased.
#    - `quantity`: The number of units purchased.
#    - `order_date`: The date the order was placed, defaulting to the current date.

# Relationships:
# - The `user_id` column in the `orders` table establishes a one-to-many relationship with the `id` column in the `users` table. Each customer can place multiple orders.

# Business Use:
# - The `users` table is primarily used for customer management and tracking.
# - The `orders` table is used to analyze purchasing patterns, inventory needs, and revenue generation.

# Example Use Cases:
# - Finding the total number of orders placed by a specific customer.
# - Determining the most popular product among all customers.
# - Calculating the average age of customers placing orders.
# """
# )


# # 4. Add an example SQL query for retrieving data
# vn.train(
#     sql="""
#     SELECT
#         users.name AS customer_name,
#         orders.product_name,
#         orders.quantity,
#         orders.order_date
#     FROM orders
#     INNER JOIN users ON orders.user_id = users.id
#     WHERE users.name = 'Naruto Uzumaki';
# """
# )


# vn.train(
#     sql="""
#     SELECT
#         orders.order_id,
#         orders.product_name,
#         orders.quantity,
#         orders.order_date
#     FROM orders
#     INNER JOIN users ON orders.user_id = users.id
#     WHERE users.name = 'Sasuke Uchiha';
# """
# )

# vn.train(
#     sql="""
#     SELECT
#         product_name,
#         SUM(quantity) AS total_quantity
#     FROM orders
#     GROUP BY product_name
#     ORDER BY total_quantity DESC
#     LIMIT 1;
# """
# )

# vn.train(
#     sql="""
#     SELECT
#         users.name AS customer_name,
#         COUNT(orders.order_id) AS total_orders
#     FROM orders
#     INNER JOIN users ON orders.user_id = users.id
#     GROUP BY users.name
#     ORDER BY total_orders DESC;
# """
# )

# vn.train(
#     sql="""
#     SELECT
#         orders.product_name,
#         SUM(orders.quantity * CASE
#             WHEN orders.product_name = 'Kunai' THEN 15
#             WHEN orders.product_name = 'Shuriken' THEN 10
#             WHEN orders.product_name = 'Medical Kit' THEN 50
#             ELSE 20
#         END) AS total_revenue
#     FROM orders
#     GROUP BY orders.product_name;
# """
# )

# vn.train(
#     sql="""
#     SELECT
#         users.name AS customer_name,
#         COUNT(orders.order_id) AS total_orders
#     FROM orders
#     INNER JOIN users ON orders.user_id = users.id
#     GROUP BY users.name
#     HAVING total_orders > 1;
# """
# )

# vn.train(
#     sql="""
#     SELECT
#         orders.product_name,
#         orders.quantity,
#         orders.order_date
#     FROM orders
#     WHERE orders.order_date >= CURDATE() - INTERVAL 7 DAY;
# """
# )

# vn.train(
#     sql="""
#     SELECT
#         AVG(users.age) AS average_age
#     FROM orders
#     INNER JOIN users ON orders.user_id = users.id;
# """
# )


# vn.train(
#     sql="""
#     SELECT
#         users.name AS customer_name,
#         orders.product_name,
#         SUM(orders.quantity) AS total_quantity
#     FROM orders
#     INNER JOIN users ON orders.user_id = users.id
#     GROUP BY users.name, orders.product_name
#     ORDER BY users.name;
# """
# )


# Inspect the training data to confirm it is correctly set up
training_data = vn.get_training_data()
print(training_data)

# If there is obsolete or incorrect training data, you can remove it
# Example: Remove training data with a specific ID
# vn.remove_training_data(id='1-ddl')

from vanna.flask import VannaFlaskApp

app = VannaFlaskApp(vn, allow_llm_to_see_data=False)
app.run()
