import sqlite3
from werkzeug.security import generate_password_hash

DB_FILE = "hotel_booking.db"

def create_admin(name, email, phone, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    hashed = generate_password_hash(password)

    try:
        c.execute("""
            INSERT INTO customers (name, email, phone, password, is_admin)
            VALUES (?, ?, ?, ?, ?)
        """, (name, email.lower(), phone, hashed, 1))
        conn.commit()
        print("Admin created successfully!")
    except sqlite3.IntegrityError:
        print("Error: Email already exists.")
    finally:
        conn.close()


if __name__ == "__main__":
    print("=== Create Admin User ===")
    name = input("Enter admin name: ")
    email = input("Enter admin email: ")
    phone = input("Enter admin phone: ")
    password = input("Enter admin password: ")

    create_admin(name, email, phone, password)
