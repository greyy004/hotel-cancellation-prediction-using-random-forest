# app.py
import os
import sqlite3
import pickle
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, redirect, url_for, flash, session
import pandas as pd

# -----------------------
# CONFIG
# -----------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "hotel_booking.db")

# model files (saved by your training script)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.pkl")

# file upload config
UPLOAD_FOLDER = os.path.join("static", "uploads", "rooms")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -----------------------
# DATABASE INITIALIZATION
# -----------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS room_types (
            room_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_type_name TEXT UNIQUE NOT NULL,
            description TEXT,
            price_per_night INTEGER,
            image_path TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rooms (
            room_id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_number TEXT UNIQUE NOT NULL,
            room_type_id INTEGER NOT NULL,
            price_per_night INTEGER,
            FOREIGN KEY(price_per_night) REFERENCES room_types(price_per_night),
            FOREIGN KEY(room_type_id) REFERENCES room_types(room_type_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meal_plans (
            meal_plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
            meal_plan_name TEXT UNIQUE NOT NULL,
            image_path TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_segments (
            market_segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_name TEXT UNIQUE NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            booking_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            room_id INTEGER NOT NULL,
            meal_plan_id INTEGER NOT NULL,
            market_segment_id INTEGER NOT NULL,
            booking_status TEXT CHECK(booking_status IN ('Canceled','Not_Canceled')) NOT NULL,
            no_of_adults INTEGER NOT NULL,
            no_of_children INTEGER NOT NULL,
            no_of_weekend_nights INTEGER NOT NULL,
            no_of_week_nights INTEGER NOT NULL,
            required_car_parking_space INTEGER DEFAULT 0,
            lead_time INTEGER NOT NULL,
            arrival_year INTEGER NOT NULL,
            arrival_month INTEGER NOT NULL,
            arrival_date INTEGER NOT NULL,
            repeated_guest INTEGER DEFAULT 0,
            no_of_previous_cancellations INTEGER DEFAULT 0,
            no_of_previous_bookings_not_canceled INTEGER DEFAULT 0,
            avg_price_per_room REAL NOT NULL,
            no_of_special_requests INTEGER DEFAULT 0,
            total_nights INTEGER,
            total_guests INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY(room_id) REFERENCES rooms(room_id),
            FOREIGN KEY(meal_plan_id) REFERENCES meal_plans(meal_plan_id),
            FOREIGN KEY(market_segment_id) REFERENCES market_segments(market_segment_id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -----------------------
# HELPERS
# -----------------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login first.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("is_admin"):
            flash("Admin access required.", "danger")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# -----------------------
# LOAD MODEL & ENCODERS
# -----------------------
rf_model = None
encoders = {}
feature_cols = [
    'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
    'required_car_parking_space', 'lead_time', 'arrival_year', 'arrival_month', 'arrival_date',
    'repeated_guest', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
    'avg_price_per_room', 'no_of_special_requests', 'type_of_meal_plan_encoded',
    'room_type_reserved_encoded', 'market_segment_type_encoded', 'total_nights', 'total_guests'
]

# Try loading trained artifacts if available
if os.path.exists(RF_MODEL_PATH):
    try:
        with open(RF_MODEL_PATH, "rb") as f:
            rf_model = pickle.load(f)
    except Exception as e:
        print("Could not load RF model:", e)

if os.path.exists(ENCODERS_PATH):
    try:
        with open(ENCODERS_PATH, "rb") as f:
            encoders = pickle.load(f)
    except Exception as e:
        print("Could not load encoders:", e)

if os.path.exists(FEATURE_COLS_PATH):
    try:
        with open(FEATURE_COLS_PATH, "rb") as f:
            feature_cols = pickle.load(f)
    except Exception as e:
        print("Could not load feature columns:", e)

# -----------------------
# DB->MODEL MAPPINGS (EDIT to match your DB values)
# -----------------------
# NOTE: these map the human-friendly DB names to the exact category strings
# that your LabelEncoders were trained on (e.g. "Meal Plan 1", "Room_Type 1", "Online").
# Update these dictionaries to reflect your actual DB values.
MEAL_MAP = {
    "Breakfast Only": "Meal Plan 1",
    "Full Board": "Meal Plan 2",
    "Half Board": "Meal Plan 3",
    "No Meal": "Not Selected",
}
ROOM_MAP = {
    "Standard": "Room_Type 1",
    "Deluxe": "Room_Type 2",
    "Executive": "Room_Type 3",
    "Family Suite": "Room_Type 4",
    "Presidential Suite": "Room_Type 5",
    "Single": "Room_Type 6",
    "Double": "Room_Type 7",
}
SEGMENT_MAP = {
    "Online": "Online",
    "Offline": "Offline",
    "Corporate": "Corporate",
    "Airline Guest": "Aviation",
    "Complementary": "Complementary",
    # add additional mappings
}

def map_and_encode(db_value, mapping_dict, encoder, default_model_cat=None):
    """
    Map a DB category value to the model's category string and return its integer encoding.
    Returns -1 if unknown or encoder not available.
    """
    if db_value is None:
        return -1

    # direct mapping
    model_cat = mapping_dict.get(db_value)

    # fallback to case-insensitive match of keys
    if model_cat is None:
        db_lower = str(db_value).strip().lower()
        for k, v in mapping_dict.items():
            if isinstance(k, str) and k.strip().lower() == db_lower:
                model_cat = v
                break

    # fallback to a default model category if provided
    if model_cat is None:
        model_cat = default_model_cat

    # final encode using encoder classes if available
    try:
        if model_cat is not None and encoder is not None and hasattr(encoder, "classes_"):
            if model_cat in encoder.classes_:
                return int(encoder.transform([model_cat])[0])
            # Try approximate match: if model_cat not found, attempt to find a classes_ entry that contains model_cat token
            mc_lower = str(model_cat).lower()
            for c in encoder.classes_:
                if mc_lower in str(c).lower() or str(c).lower() in mc_lower:
                    return int(encoder.transform([c])[0])
    except Exception:
        pass

    return -1

# -----------------------
# ROUTES
# -----------------------
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password_hash = generate_password_hash(request.form["password"])
        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO customers (name, email, password) VALUES (?, ?, ?)",
                (name, email, password_hash)
            )
            conn.commit()
            flash("Registration successful! You can now login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already exists!", "danger")
        finally:
            conn.close()
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM customers WHERE email=?", (email,)).fetchone()
        conn.close()
        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["customer_id"]
            session["is_admin"] = bool(user["is_admin"])
            flash("Login successful!", "success")
            return redirect(url_for("admin_dashboard" if user["is_admin"] else "user_dashboard"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out", "success")
    return redirect(url_for("landing"))

@app.route("/user_dashboard")
@login_required
def user_dashboard():
    if session.get("is_admin"):
        flash("User access required.", "danger")
        return redirect(url_for("admin_dashboard"))
    # Add logic to fetch user-specific data here
    available_rooms = []
    conn = get_db_connection()
    available_rooms = conn.execute("""
        SELECT t1.*, t2.image_path, t2.room_type_name, t2.description
FROM rooms t1
LEFT JOIN room_types t2 ON t1.room_type_id = t2.room_type_id;""").fetchall() 
    conn.close()
    return render_template("user_dashboard.html", available_rooms=available_rooms)


# -----------------------
# BOOK ROOM
# -----------------------
@app.route("/book_room/<int:room_id>", methods=["GET", "POST"])
@login_required
def book_room(room_id):
    conn = get_db_connection()
    
    # Fetch room details
    room = conn.execute("""
        SELECT r.room_id, r.room_number, r.price_per_night, t.room_type_name, t.image_path, t.description
        FROM rooms r
        JOIN room_types t ON r.room_type_id = t.room_type_id
        WHERE r.room_id=?
    """, (room_id,)).fetchone()
    
    if not room:
        flash("Room not found!", "danger")
        return redirect(url_for("user_dashboard"))

    meal_plans = conn.execute("SELECT * FROM meal_plans").fetchall()
    segments = conn.execute("SELECT * FROM market_segments").fetchall()

    customer_id = session["user_id"]

    # === AUTOMATICALLY GET REAL CUSTOMER HISTORY ===
    prev_bookings = conn.execute("""
        SELECT booking_status 
        FROM bookings 
        WHERE customer_id = ?
        ORDER BY created_at DESC
    """, (customer_id,)).fetchall()

    total_previous_bookings = len(prev_bookings)
    
    # Count how many were canceled
    previous_cancellations = sum(1 for b in prev_bookings if b["booking_status"] == "Canceled")
    
    # Count how many were NOT canceled (i.e. completed/stayed)
    previous_successful = sum(1 for b in prev_bookings if b["booking_status"] == "Not_Canceled")
    
    # Is this person has booked before?
    repeated_guest = 1 if total_previous_bookings > 0 else 0

    if request.method == "POST":
        try:
            no_of_adults = int(request.form["no_of_adults"])
            no_of_children = int(request.form["no_of_children"])
            no_of_weekend_nights = int(request.form["no_of_weekend_nights"])
            no_of_week_nights = int(request.form["no_of_week_nights"])
            lead_time = int(request.form["lead_time"])
            arrival_year = int(request.form["arrival_year"])
            arrival_month = int(request.form["arrival_month"])
            arrival_date = int(request.form["arrival_date"])
            
            avg_price = float(room["price_per_night"])  # Use actual room price
            special_requests = int(request.form.get("no_of_special_requests", 0))
            required_parking = int(request.form.get("required_car_parking_space", 0))
            meal_plan_id = int(request.form["meal_plan_id"])
            segment_id = int(request.form["market_segment_id"])

            total_nights = no_of_weekend_nights + no_of_week_nights
            total_guests = no_of_adults + no_of_children

            conn.execute("""
                INSERT INTO bookings (
                    customer_id, room_id, meal_plan_id, market_segment_id, booking_status,
                    no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights,
                    lead_time, arrival_year, arrival_month, arrival_date,
                    avg_price_per_room, no_of_special_requests,
                    repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
                    required_car_parking_space, total_nights, total_guests
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                customer_id, room["room_id"], meal_plan_id, segment_id, "Not_Canceled",
                no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights,
                lead_time, arrival_year, arrival_month, arrival_date,
                avg_price, special_requests,
                repeated_guest, previous_cancellations, previous_successful,
                required_parking, total_nights, total_guests
            ))
            conn.commit()
            flash(f"Room {room['room_number']} booked successfully!", "success")
            return redirect(url_for("user_dashboard"))

        except Exception as e:
            conn.rollback()
            flash("Booking failed. Please try again.", "danger")
            print(e)

    conn.close()
    return render_template(
        "book_room.html", 
        room=room, 
        meal_plans=meal_plans, 
        segments=segments,
        repeated_guest=repeated_guest,
        previous_cancellations=previous_cancellations,
        previous_successful=previous_successful
    )
@app.route("/my_bookings")
@login_required
def my_bookings():
    user_id = session["user_id"]
    conn = get_db_connection()

    # Fetch user's bookings with room type info and image path
    bookings = conn.execute("""
        SELECT b.booking_id,
               r.room_number,
               t.room_type_name,
               s.segment_name,
               b.no_of_adults,
               b.no_of_children,
               b.total_nights,
               b.total_guests,
               b.booking_status,
               b.created_at,
               b.avg_price_per_room,
               b.total_nights,
               t.image_path
        FROM bookings b
        LEFT JOIN rooms r ON b.room_id = r.room_id
        LEFT JOIN room_types t ON r.room_type_id = t.room_type_id
        LEFT JOIN meal_plans m ON b.meal_plan_id = m.meal_plan_id
        LEFT JOIN market_segments s ON b.market_segment_id = s.market_segment_id
        WHERE b.customer_id = ?
        ORDER BY b.created_at DESC
    """, (user_id,)).fetchall()

    conn.close()
    return render_template("my_bookings.html", bookings=bookings)


# Cancel booking route
@app.route("/cancel_booking/<int:booking_id>", methods=["POST"])
@login_required
def cancel_booking(booking_id):
    user_id = session["user_id"]
    conn = get_db_connection()

    # Check if booking exists and belongs to the logged-in user
    booking = conn.execute(
        "SELECT * FROM bookings WHERE booking_id = ? AND customer_id = ?",
        (booking_id, user_id)
    ).fetchone()

    if not booking:
        flash("Booking not found or you are not authorized to cancel it.", "danger")
        conn.close()
        return redirect(url_for("my_bookings"))

    if booking["booking_status"] == "Canceled":
        flash("This booking is already canceled.", "info")
        conn.close()
        return redirect(url_for("my_bookings"))

    # Update booking status to Canceled
    conn.execute(
        "UPDATE bookings SET booking_status = 'Canceled', updated_at = CURRENT_TIMESTAMP WHERE booking_id = ?",
        (booking_id,)
    )
    conn.commit()
    conn.close()

    flash("Your booking has been canceled successfully.", "success")
    return redirect(url_for("my_bookings"))

# -----------------------
# DASHBOARDS
# -----------------------
@app.route("/admin_dashboard")
@login_required
def admin_dashboard():
    if not session.get("is_admin"): 
        flash("Admin access required.", "danger")
        return redirect(url_for("login"))

    conn = get_db_connection()

    # -----------------------
    # Stats
    # -----------------------
    total_bookings = conn.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
    available_rooms = conn.execute("""
        SELECT COUNT(*) FROM rooms
    """).fetchone()[0]
    total_meal_plans = conn.execute("SELECT COUNT(*) FROM meal_plans").fetchone()[0]
    total_users = conn.execute("SELECT COUNT(*) FROM customers WHERE is_admin=0").fetchone()[0]

    # -----------------------
    # Recent bookings
    # -----------------------
    recent_bookings = conn.execute("""
    SELECT b.booking_id, 
           c.name as customer_name, 
           COALESCE(r.room_number, 'Deleted Room') as room_number,
           COALESCE(t.room_type_name, 'Unknown Type') as room_type_name,
           m.meal_plan_name, 
           s.segment_name, 
           b.no_of_adults, 
           b.no_of_children,
           b.total_nights, 
           b.booking_status, 
           b.created_at
    FROM bookings b
    JOIN customers c ON b.customer_id = c.customer_id
    LEFT JOIN rooms r ON b.room_id = r.room_id
    LEFT JOIN room_types t ON r.room_type_id = t.room_type_id
    JOIN meal_plans m ON b.meal_plan_id = m.meal_plan_id
    JOIN market_segments s ON b.market_segment_id = s.market_segment_id
    ORDER BY b.created_at DESC
    LIMIT 5
""").fetchall()

    # -----------------------
    # Room types
    # -----------------------
    room_types = conn.execute("""
        SELECT room_type_id, room_type_name, description, price_per_night
        FROM room_types
        ORDER BY room_type_id
    """).fetchall()


    conn.close()

    return render_template(
        "admin_dashboard.html",
        total_bookings=total_bookings,
        available_rooms=available_rooms,
        total_meal_plans=total_meal_plans,
        total_users=total_users,
        recent_bookings=recent_bookings,
        room_types=room_types
    )

# -----------------------
# ADMIN: ROOM TYPES, ROOMS, MEAL PLANS, SEGMENTS
# -----------------------
@app.route("/admin/room_types", methods=["GET", "POST"])
@admin_required
def manage_room_types():
    conn = get_db_connection()
    if request.method == "POST":
        name = request.form["room_type_name"]
        desc = request.form.get("description", "")
        price = request.form.get("price_per_night", 0)
        file = request.files.get("image_file")
        img_filename = None
        if file and allowed_file(file.filename):
            img_filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], img_filename))
        try:
            conn.execute("INSERT INTO room_types (room_type_name,description,price_per_night,image_path) VALUES (?,?,?,?)",
                         (name, desc, price, img_filename))
            conn.commit()
            flash("Room type added!", "success")
        except sqlite3.IntegrityError:
            flash("Room type exists!", "danger")
    room_types = conn.execute("SELECT * FROM room_types").fetchall()
    conn.close()
    return render_template("manage_room_types.html", room_types=room_types)


# -----------------------
# ADMIN: ROOMS
# -----------------------
@app.route("/admin/rooms", methods=["GET", "POST"])
@admin_required
def manage_rooms():
    conn = get_db_connection()
    room_types = conn.execute("SELECT * FROM room_types").fetchall()

    if request.method == "POST":
        room_number = request.form.get("room_number")
        room_type_id = request.form.get("room_type_id")

        if not room_number or not room_type_id:
            flash("Room number and type are required.", "danger")
        else:
            try:
                room_type_id = int(room_type_id)
            except ValueError:
                flash("Invalid room type selected.", "danger")
            else:
                # Fetch default price from room_types
                row = conn.execute(
                    "SELECT price_per_night FROM room_types WHERE room_type_id=?",
                    (room_type_id,)
                ).fetchone()
                default_price = row[0] if row else 0.0

                # Get admin input price or fallback to default
                price_input = request.form.get("price_per_night")
                try:
                    price = float(price_input) if price_input else default_price
                except ValueError:
                    price = default_price  # fallback if conversion fails

                # Insert the room
                try:
                    conn.execute(
                        "INSERT INTO rooms (room_number, room_type_id, price_per_night) VALUES (?, ?, ?)",
                        (room_number, room_type_id, price)
                    )
                    conn.commit()
                    flash("Room added successfully!", "success")
                except sqlite3.IntegrityError:
                    flash("A room with this number already exists!", "danger")

    # Fetch all rooms for display
    rooms = conn.execute("""
        SELECT r.room_id, r.room_number, t.room_type_name, t.image_path, r.price_per_night,
               t.price_per_night AS type_default_price
        FROM rooms r
        JOIN room_types t ON r.room_type_id = t.room_type_id
        ORDER BY r.room_number
    """).fetchall()

    conn.close()
    return render_template("manage_rooms.html", rooms=rooms, room_types=room_types)

# -----------------------
# ADMIN: MEAL PLANS,
# -----------------------
# ================= MANAGE MEAL PLANS =================
@app.route("/admin/manage_meal_plans", methods=["GET", "POST"])
@admin_required
def manage_meal_plans():
    conn = get_db_connection()
    cursor = conn.cursor()

    # ----------- Add New Meal Plan -----------
    if request.method == "POST":
        meal_plan_name = request.form["meal_plan_name"]
        image_file = request.files.get("image_file")

        image_filename = None

        # Save Image if uploaded
        if image_file and image_file.filename != "":
            image_filename = secure_filename(image_file.filename)
            image_path = os.path.join(UPLOAD_FOLDER, image_filename)
            image_file.save(image_path)

        cursor.execute("""
            INSERT INTO meal_plans (meal_plan_name, image_path)
            VALUES (?, ?)
        """, (meal_plan_name, image_filename))

        conn.commit()
        flash("Meal plan added!", "success")
        return redirect(url_for("manage_meal_plans"))

    # ----------- Fetch Meal Plans -----------
    meal_plans = cursor.execute("SELECT * FROM meal_plans").fetchall()

    conn.close()
    return render_template("manage_meal_plans.html", meal_plans=meal_plans)


@app.route("/admin/delete_meal_plan/<int:meal_id>")
@admin_required
def delete_meal_plan(meal_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get image filename for deletion
    row = cursor.execute(
        "SELECT image_path FROM meal_plans WHERE meal_plan_id = ?",
        (meal_id,)
    ).fetchone()

    # Delete image file
    if row and row["image_path"]:
        img_path = os.path.join(UPLOAD_FOLDER, row["image_path"])
        if os.path.exists(img_path):
            os.remove(img_path)

    # Delete DB row
    cursor.execute(
        "DELETE FROM meal_plans WHERE meal_plan_id = ?",
        (meal_id,)
    )

    conn.commit()
    conn.close()

    flash("Meal plan deleted!", "success")
    return redirect(url_for("manage_meal_plans"))


# -----------------------# ADMIN: MARKET SEGMENTS
# -----------------------

@app.route("/admin/market_segments", methods=["GET", "POST"])
@admin_required
def manage_market_segments():
    conn = get_db_connection()
    if request.method == "POST":
        name = request.form["segment_name"]
        try:
            conn.execute("INSERT INTO market_segments (segment_name) VALUES (?)", (name,))
            conn.commit()
            flash("Segment added!", "success")
        except sqlite3.IntegrityError:
            flash("Segment exists!", "danger")
    segments = conn.execute("SELECT * FROM market_segments").fetchall()
    conn.close()
    return render_template("manage_market_segments.html", segments=segments)

# -----------------------
# ADMIN: DELETE ROOM
# -----------------------
@app.route("/admin/rooms/delete/<int:room_id>", methods=["POST"])
@admin_required
def delete_room(room_id):
    conn = get_db_connection()

    # Check if room exists
    room = conn.execute("SELECT * FROM rooms WHERE room_id = ?", (room_id,)).fetchone()
    if room is None:
        conn.close()
        flash("Room not found!", "danger")
        return redirect(url_for("manage_rooms"))

    # Delete room
    conn.execute("DELETE FROM rooms WHERE room_id = ?", (room_id,))
    conn.commit()
    conn.close()

    flash("Room deleted successfully!", "success")
    return redirect(url_for("manage_rooms"))

# -----------------------
# ADMIN: VIEW BOOKINGS WITH PREDICTIONS
# -----------------------
@app.route("/admin/bookings")
@admin_required
def admin_view_bookings():
    conn = get_db_connection()
    bookings = conn.execute("""
        SELECT b.*, 
               c.name as customer_name, 
               COALESCE(r.room_number, 'Deleted Room') as room_number,
               COALESCE(t.room_type_name, 'Unknown Room Type') as room_type_name,
               m.meal_plan_name, 
               s.segment_name
        FROM bookings b
        JOIN customers c ON b.customer_id = c.customer_id
        LEFT JOIN rooms r ON b.room_id = r.room_id
        LEFT JOIN room_types t ON r.room_type_id = t.room_type_id
        JOIN meal_plans m ON b.meal_plan_id = m.meal_plan_id
        JOIN market_segments s ON b.market_segment_id = s.market_segment_id
        ORDER BY b.created_at DESC
    """).fetchall()

    booking_preds = []

    # === Your prediction code stays 100% unchanged ===
    meal_encoder = encoders.get("type_of_meal_plan") or encoders.get("type_of_meal_plan_encoded")
    room_encoder = encoders.get("room_type_reserved") or encoders.get("room_type_reserved_encoded")
    seg_encoder = encoders.get("market_segment_type") or encoders.get("market_segment_type_encoded")

    for b in bookings:
        meal_enc = map_and_encode(
            b["meal_plan_name"],
            MEAL_MAP,
            meal_encoder,
            default_model_cat="Not Selected"
        )
        room_enc = map_and_encode(
            b["room_type_name"],        # ← This will be 'Unknown Room Type' if room deleted
            ROOM_MAP,
            room_encoder,
            default_model_cat="Room_Type 1"
        )
        seg_enc = map_and_encode(
            b["segment_name"],
            SEGMENT_MAP,
            seg_encoder,
            default_model_cat="Offline"
        )

        df_input = pd.DataFrame([{
            "no_of_adults": b["no_of_adults"],
            "no_of_children": b["no_of_children"],
            "no_of_weekend_nights": b["no_of_weekend_nights"],
            "no_of_week_nights": b["no_of_week_nights"],
            "required_car_parking_space": b["required_car_parking_space"],
            "lead_time": b["lead_time"],
            "arrival_year": b["arrival_year"],
            "arrival_month": b["arrival_month"],
            "arrival_date": b["arrival_date"],
            "repeated_guest": b["repeated_guest"],
            "no_of_previous_cancellations": b["no_of_previous_cancellations"],
            "no_of_previous_bookings_not_canceled": b["no_of_previous_bookings_not_canceled"],
            "avg_price_per_room": b["avg_price_per_room"],
            "no_of_special_requests": b["no_of_special_requests"],
            "type_of_meal_plan_encoded": meal_enc,
            "room_type_reserved_encoded": room_enc,
            "market_segment_type_encoded": seg_enc,
            "total_nights": b["total_nights"] if b["total_nights"] is not None else (b["no_of_weekend_nights"] + b["no_of_week_nights"]),
            "total_guests": b["total_guests"] if b["total_guests"] is not None else (b["no_of_adults"] + b["no_of_children"])
        }])

        df_input = df_input.reindex(columns=feature_cols, fill_value=0)
        prob = rf_model.predict_proba(df_input)[0, 1] if rf_model is not None else 0.0

        booking_preds.append({
            "booking_id": b["booking_id"],
            "cancellation_probability": round(prob, 3),
            "prediction": "Likely to Cancel" if prob > 0.5 else "Likely to NOT Cancel",
            "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low",
        })

    conn.close()
    return render_template("admin_bookings.html", bookings=bookings, bookings_combined=zip(bookings, booking_preds))

# -----------------------
# ADMIN: VIEW BOOKING FEATURES
# -----------------------
@app.route("/admin/booking_features/<int:booking_id>")
@admin_required
def admin_view_booking_features(booking_id):
    conn = get_db_connection()
    b = conn.execute("""
        SELECT b.*, c.name as customer_name, r.room_number, t.room_type_name,
               m.meal_plan_name, s.segment_name
        FROM bookings b
        JOIN customers c ON b.customer_id = c.customer_id
        JOIN rooms r ON b.room_id = r.room_id
        JOIN room_types t ON r.room_type_id = t.room_type_id
        JOIN meal_plans m ON b.meal_plan_id = m.meal_plan_id
        JOIN market_segments s ON b.market_segment_id = s.market_segment_id
        WHERE b.booking_id=?
    """, (booking_id,)).fetchone()
    conn.close()

    if not b:
        flash("Booking not found!", "danger")
        return redirect(url_for("admin_view_bookings"))

    # Encode features
    meal_encoder = encoders.get("type_of_meal_plan") or encoders.get("type_of_meal_plan_encoded")
    room_encoder = encoders.get("room_type_reserved") or encoders.get("room_type_reserved_encoded")
    seg_encoder = encoders.get("market_segment_type") or encoders.get("market_segment_type_encoded")

    meal_enc = map_and_encode(b["meal_plan_name"], MEAL_MAP, meal_encoder, "Not Selected")
    room_enc = map_and_encode(b["room_type_name"], ROOM_MAP, room_encoder, "Room_Type 1")
    seg_enc = map_and_encode(b["segment_name"], SEGMENT_MAP, seg_encoder, "Offline")

    features = {
        "no_of_adults": b["no_of_adults"],
        "no_of_children": b["no_of_children"],
        "no_of_weekend_nights": b["no_of_weekend_nights"],
        "no_of_week_nights": b["no_of_week_nights"],
        "required_car_parking_space": b["required_car_parking_space"],
        "lead_time": b["lead_time"],
        "arrival_year": b["arrival_year"],
        "arrival_month": b["arrival_month"],
        "arrival_date": b["arrival_date"],
        "repeated_guest": b["repeated_guest"],
        "no_of_previous_cancellations": b["no_of_previous_cancellations"],
        "no_of_previous_bookings_not_canceled": b["no_of_previous_bookings_not_canceled"],
        "avg_price_per_room": b["avg_price_per_room"],
        "no_of_special_requests": b["no_of_special_requests"],
        "type_of_meal_plan_encoded": meal_enc,
        "room_type_reserved_encoded": room_enc,
        "market_segment_type_encoded": seg_enc,
        "total_nights": b["total_nights"] if b["total_nights"] else (b["no_of_weekend_nights"] + b["no_of_week_nights"]),
        "total_guests": b["no_of_adults"] + b["no_of_children"]
    }

    return render_template("admin_booking_features.html", booking=b, features=features)


# -----------------------
# RUN
# -----------------------
if __name__ == "__main__":
    # If running directly, use debug=True for development only
    app.run(debug=True)
