from sqlalchemy import create_engine, inspect

# Create a SQLAlchemy engine to connect to your database
engine = create_engine('sqlite:///C:/Users/PARTH SARDA/Downloads/sqlite-tools-win32-x86-3430100/MyDatabases/testdb.db')

# Create an inspector
inspector = inspect(engine)

# Replace 'your_table_name' with the name of the table you want to check
table_name = 'students'

# Check if the table exists
if inspector.has_table(table_name):
    print(f"The table '{table_name}' exists in the database.")
else:
    print(f"The table '{table_name}' does not exist in the database.")
