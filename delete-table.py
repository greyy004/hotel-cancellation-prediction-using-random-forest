# delete_table.py

import sqlite3

# Replace with your database file
DB_FILE = "hotel_booking.db"
TABLE_NAME = "meal_plans"  # Replace with the table you want to delete

def delete_table(db_file, table_name):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Warning: This will permanently delete the table
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        
        print(f"Table '{table_name}' has been deleted successfully.")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    delete_table(DB_FILE, TABLE_NAME)
