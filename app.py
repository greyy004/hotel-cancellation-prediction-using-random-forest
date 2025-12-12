# app.py - COMPLETE VERSION WITH KHALTI PAYMENT INTEGRATION
import os
import sqlite3
import pickle
import requests
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify

import pandas as pd

# -----------------------
# CONFIG
# -----------------------
app = Flask(__name__)
# Static secret key set per user request (replace in production)
app.secret_key = "2f6e84c4f8584b3cb6b3d6a1f059d4c7ca3a965a0e6a4d1f9a6ef1c4f2c1b8a2"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "hotel_booking.db")

# Model files
MODEL_DIR = os.path.join(BASE_DIR, "model_files")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.pkl")

# File upload config
UPLOAD_FOLDER = os.path.join("static", "uploads", "rooms")
MENU_PLAN_UPLOAD_FOLDER = os.path.join("static", "uploads", "menu_plans")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MENU_PLAN_UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MENU_PLAN_UPLOAD_FOLDER"] = MENU_PLAN_UPLOAD_FOLDER

# ========================================
# KHALTI CONFIGURATION
# ========================================
# Static keys set per user request (replace in production)
KHALTI_SECRET_KEY = "089cde588c244d8da5be025f0d6fbf7c"
KHALTI_PUBLIC_KEY = "1947ad7b453940f7a8ec8291e0a22873"
KHALTI_GATEWAY_URL = "https://dev.khalti.com/api/v2"

app.config['KHALTI_SECRET_KEY'] = KHALTI_SECRET_KEY
app.config['KHALTI_PUBLIC_KEY'] = KHALTI_PUBLIC_KEY

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
            phone TEXT,
            address TEXT,
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
            FOREIGN KEY(room_type_id) REFERENCES room_types(room_type_id) ON DELETE CASCADE
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
    # Enforce foreign key constraints on every connection
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def compute_total_nights_from_row(row):
    """Compute total nights from a DB row, falling back to component nights."""
    total_nights = row["total_nights"]
    # Treat zero/negative as missing to avoid hiding booked ranges
    if total_nights is not None and total_nights > 0:
        return total_nights
    return (row["no_of_weekend_nights"] or 0) + (row["no_of_week_nights"] or 0)

def booking_window_from_payload(data):
    """Derive check-in/check-out dates from booking payload."""
    total_nights = int(data.get("total_nights") or 0)
    checkin = datetime(
        int(data["arrival_year"]),
        int(data["arrival_month"]),
        int(data["arrival_date"])
    ).date()
    checkout = checkin + timedelta(days=total_nights)
    return checkin, checkout, total_nights

def booking_window_from_row(row):
    """Derive check-in/check-out dates from a bookings table row."""
    total_nights = compute_total_nights_from_row(row)
    checkin = datetime(
        row["arrival_year"],
        row["arrival_month"],
        row["arrival_date"]
    ).date()
    checkout = checkin + timedelta(days=total_nights)
    return checkin, checkout, total_nights

def is_room_available(room_id, checkin, checkout):
    """Check for overlapping active bookings for the same room."""
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT room_id, arrival_year, arrival_month, arrival_date,
               total_nights, no_of_weekend_nights, no_of_week_nights, booking_status
        FROM bookings
        WHERE room_id = ? AND booking_status = 'Not_Canceled'
    """, (room_id,)).fetchall()
    conn.close()

    for r in rows:
        existing_checkin, existing_checkout, _ = booking_window_from_row(r)
        if checkin < existing_checkout and checkout > existing_checkin:
            return False
    return True


@app.route("/api/rooms/<int:room_id>/unavailable", methods=["GET"])
def room_unavailable_ranges(room_id):
    """Return booked date ranges for a room (exclusive checkout) for client-side blocking."""
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT arrival_year, arrival_month, arrival_date,
               total_nights, no_of_weekend_nights, no_of_week_nights
        FROM bookings
        WHERE room_id = ? AND booking_status = 'Not_Canceled'
    """, (room_id,)).fetchall()
    conn.close()

    ranges = []
    for r in rows:
        start, end, total_nights = booking_window_from_row(r)
        if total_nights <= 0:
            continue
        ranges.append({
            "start": start.isoformat(),
            "end": end.isoformat()  # checkout (exclusive)
        })
    return jsonify({"ranges": ranges})

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login first.", "warning")
            # Preserve the current URL as return URL
            next_url = request.url
            return redirect(url_for("login", next=next_url))
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

def create_booking_from_session(booking_data):
    """Create booking using data stored in session after payment"""
    conn = get_db_connection()
    try:
        customer_id = session["user_id"]
        conn.execute("""
            INSERT INTO bookings (
                customer_id, room_id, meal_plan_id, market_segment_id, booking_status,
                no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights,
                lead_time, arrival_year, arrival_month, arrival_date,
                avg_price_per_room, no_of_special_requests, required_car_parking_space,
                repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
                total_nights, total_guests
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            customer_id, booking_data['room_id'], booking_data['meal_plan_id'], 
            booking_data['market_segment_id'], "Not_Canceled",
            booking_data['no_of_adults'], booking_data['no_of_children'],
            booking_data['no_of_weekend_nights'], booking_data['no_of_week_nights'],
            booking_data['lead_time'], booking_data['arrival_year'], 
            booking_data['arrival_month'], booking_data['arrival_date'],
            booking_data['avg_price_per_room'], booking_data.get('no_of_special_requests', 0),
            booking_data.get('required_car_parking_space', 0),
            booking_data.get('repeated_guest', 0), booking_data.get('no_of_previous_cancellations', 0),
            booking_data.get('no_of_previous_bookings_not_canceled', 0),
            booking_data['total_nights'], booking_data['total_guests']
        ))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Booking creation failed: {e}")
        return False
    finally:
        conn.close()

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
# DB->MODEL MAPPINGS
# -----------------------
MEAL_MAP = {
    "Mixed": "Meal Plan 1",
    "Veg": "Meal Plan 2",
    "Non Veg": "Meal Plan 3",
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
}

def map_and_encode(db_value, mapping_dict, encoder, default_model_cat=None):
    if db_value is None:
        return -1
    model_cat = mapping_dict.get(db_value)
    if model_cat is None:
        db_lower = str(db_value).strip().lower()
        for k, v in mapping_dict.items():
            if isinstance(k, str) and k.strip().lower() == db_lower:
                model_cat = v
                break
    if model_cat is None:
        model_cat = default_model_cat
    try:
        if model_cat is not None and encoder is not None and hasattr(encoder, "classes_"):
            if model_cat in encoder.classes_:
                return int(encoder.transform([model_cat])[0])
            mc_lower = str(model_cat).lower()
            for c in encoder.classes_:
                if mc_lower in str(c).lower() or str(c).lower() in mc_lower:
                    return int(encoder.transform([c])[0])
    except Exception:
        pass
    return -1

# ========================================
# KHALTI PAYMENT ROUTES
# ========================================

@app.route("/api/khalti/create-payment", methods=["POST"])
@login_required
def create_khalti_payment():
    """Initiate Khalti ePayment"""
    data = request.get_json()
    
    if not data or 'amount' not in data or 'booking_data' not in data:
        return jsonify({"success": False, "error": "Invalid request data"}), 400

    # Validate dates and availability
    try:
        checkin, checkout, total_nights = booking_window_from_payload(data["booking_data"])
    except Exception:
        return jsonify({"success": False, "error": "Invalid date selection"}), 400
    if total_nights <= 0:
        return jsonify({"success": False, "error": "Stay must be at least 1 night"}), 400
    if not is_room_available(data["booking_data"]["room_id"], checkin, checkout):
        return jsonify({"success": False, "error": "Room unavailable for selected dates"}), 400
    
    # Convert to paisa (1 NPR = 100 paisa)
    amount = int(float(data['amount']) * 100)
    
    conn = get_db_connection()
    user = conn.execute(
        "SELECT name, email, phone FROM customers WHERE customer_id=?", 
        (session["user_id"],)
    ).fetchone()
    conn.close()
    
    # Generate unique purchase order ID
    purchase_order_id = f"ORDER_{session['user_id']}_{int(datetime.now().timestamp())}"
    
    # Prepare payment payload
    payload = {
        "return_url": url_for('payment_success', _external=True),
        "website_url": url_for('landing', _external=True),
        "amount": amount,
        "purchase_order_id": purchase_order_id,
        "purchase_order_name": f"Hotel Room - {data['booking_data']['room_number']}",
        "customer_info": {
            "name": user['name'] if user else 'Guest',
            "email": user['email'] if user else 'guest@hotel.com',
            "phone": user['phone'] if user and user['phone'] else '9800000000'
        }
    }
    
    headers = {
        "Authorization": f"Key {KHALTI_SECRET_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{KHALTI_GATEWAY_URL}/epayment/initiate/",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        print(f"Khalti initiate response: {response.status_code}")
        print(f"Response body: {response.text}")
        
        if response.status_code == 200:
            payment_data = response.json()
            
            # Store booking data in session
            session['pending_booking'] = data['booking_data']
            session['pending_amount'] = data['amount']
            session['pending_pidx'] = payment_data.get('pidx')
            session['purchase_order_id'] = purchase_order_id
            
            return jsonify({
                "success": True,
                "payment_url": payment_data.get('payment_url'),
                "pidx": payment_data.get('pidx')
            })
        else:
            error_data = response.json() if response.text else {}
            error_detail = error_data.get('detail', error_data.get('error_message', 'Unknown error'))
            print(f"Khalti initiate failed: {response.status_code} - {error_detail}")
            return jsonify({
                "success": False, 
                "error": f"Payment initiation failed: {error_detail}"
            }), 400
            
    except requests.exceptions.RequestException as e:
        print(f"Khalti API request error: {e}")
        return jsonify({
            "success": False, 
            "error": "Unable to connect to payment gateway"
        }), 500
    except Exception as e:
        print(f"Unexpected error in payment initiation: {e}")
        return jsonify({
            "success": False, 
            "error": "Payment initiation failed"
        }), 500


@app.route("/payment-success")
@login_required
def payment_success():
    """Handle Khalti payment return and verify transaction"""
    
    pidx = request.args.get('pidx')
    txnId = request.args.get('txnId')
    amount = request.args.get('amount')
    
    print(f"Payment callback: pidx={pidx}, txnId={txnId}, amount={amount}")
    
    if not pidx:
        flash("Invalid payment response.", "danger")
        return redirect(url_for("user_dashboard"))

    # Validate callback pidx matches the one we initiated
    if pidx != session.get('pending_pidx'):
        flash("Payment session mismatch. Please try again.", "danger")
        return redirect(url_for("user_dashboard"))
    
    headers = {
        "Authorization": f"Key {KHALTI_SECRET_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{KHALTI_GATEWAY_URL}/epayment/lookup/",
            json={"pidx": pidx},
            headers=headers,
            timeout=30
        )
        
        print(f"Khalti lookup response: {response.status_code}")
        print(f"Response body: {response.text}")
        
        if response.status_code == 200:
            payment_data = response.json()
            payment_status = payment_data.get('status', '').lower()
            lookup_purchase_order = payment_data.get('purchase_order_id')
            
            if payment_status == 'completed':
                booking_data = session.get('pending_booking')
                expected_amount = int(float(session.get('pending_amount', 0)) * 100)
                actual_amount = payment_data.get('total_amount', 0)
                expected_po = session.get('purchase_order_id')
                
                if not booking_data:
                    flash("Booking data not found. Contact support.", "danger")
                    return redirect(url_for("user_dashboard"))

                try:
                    checkin, checkout, total_nights = booking_window_from_payload(booking_data)
                    if total_nights <= 0:
                        flash("Invalid stay length. Please rebook.", "danger")
                        return redirect(url_for("user_dashboard"))
                    if not is_room_available(booking_data['room_id'], checkin, checkout):
                        flash("Selected room is no longer available for those dates.", "danger")
                        session.pop('pending_booking', None)
                        session.pop('pending_amount', None)
                        session.pop('pending_pidx', None)
                        session.pop('purchase_order_id', None)
                        return redirect(url_for("user_dashboard"))
                except Exception:
                    flash("Invalid booking dates. Please rebook.", "danger")
                    session.pop('pending_booking', None)
                    session.pop('pending_amount', None)
                    session.pop('pending_pidx', None)
                    session.pop('purchase_order_id', None)
                    return redirect(url_for("user_dashboard"))

                if expected_po and lookup_purchase_order and expected_po != lookup_purchase_order:
                    flash("Payment reference mismatch. Contact support.", "danger")
                    session.pop('pending_booking', None)
                    session.pop('pending_amount', None)
                    session.pop('pending_pidx', None)
                    session.pop('purchase_order_id', None)
                    return redirect(url_for("user_dashboard"))
                
                if abs(actual_amount - expected_amount) > 1:
                    flash(f"Payment amount mismatch. Contact support with ref: {pidx}", "danger")
                    return redirect(url_for("user_dashboard"))
                
                if create_booking_from_session(booking_data):
                    session.pop('pending_booking', None)
                    session.pop('pending_amount', None)
                    session.pop('pending_pidx', None)
                    session.pop('purchase_order_id', None)
                    
                    flash(f"Payment successful! Booking confirmed. Ref: {txnId}", "success")
                    return redirect(url_for("my_bookings"))
                else:
                    flash(f"Payment received but booking failed. Contact support: {pidx}", "danger")
                    return redirect(url_for("user_dashboard"))
            
            elif payment_status in ['pending', 'initiated', 'user_initiated']:
                flash("Payment processing. Please wait and refresh.", "info")
                return redirect(url_for("user_dashboard"))
            
            else:
                flash(f"Payment {payment_status}. Please try again.", "warning")
                session.pop('pending_booking', None)
                session.pop('pending_amount', None)
                session.pop('pending_pidx', None)
                return redirect(url_for("user_dashboard"))
        
        else:
            flash("Unable to verify payment. Contact support if charged.", "danger")
            return redirect(url_for("user_dashboard"))
            
    except Exception as e:
        print(f"Payment verification error: {e}")
        flash("Error verifying payment. Contact support if charged.", "danger")
        return redirect(url_for("user_dashboard"))


@app.route("/payment-cancel")
@login_required
def payment_cancel():
    """Handle payment cancellation"""
    session.pop('pending_booking', None)
    session.pop('pending_amount', None)
    session.pop('pending_pidx', None)
    session.pop('purchase_order_id', None)
    
    flash("Payment cancelled. You can try booking again.", "info")
    return redirect(url_for("user_dashboard"))

# ========================================
# STANDARD ROUTES (Unchanged from original)
# ========================================

@app.route("/")
def landing():
    conn = get_db_connection()
    available_rooms = conn.execute("""
        SELECT t1.*, t2.image_path, t2.room_type_name, t2.description
        FROM rooms t1
        LEFT JOIN room_types t2 ON t1.room_type_id = t2.room_type_id
    """).fetchall() 
    conn.close()
    return render_template("landing.html", available_rooms=available_rooms)

@app.route("/view_room/<int:room_id>")
def view_room(room_id):
    """Show room details - no login required"""
    conn = get_db_connection()
    room = conn.execute("""
        SELECT r.room_id, r.room_number, r.price_per_night, t.room_type_name, t.image_path, t.description
        FROM rooms r
        JOIN room_types t ON r.room_type_id = t.room_type_id
        WHERE r.room_id=?
    """, (room_id,)).fetchone()
    
    if not room:
        flash("Room not found!", "danger")
        conn.close()
        return redirect(url_for("landing"))
    
    # Fetch meal plans
    meal_plans = conn.execute("SELECT * FROM meal_plans").fetchall()
    
    # Precompute unavailable ranges for display
    booked_rows = conn.execute("""
        SELECT arrival_year, arrival_month, arrival_date,
               total_nights, no_of_weekend_nights, no_of_week_nights
        FROM bookings
        WHERE room_id = ? AND booking_status = 'Not_Canceled'
    """, (room_id,)).fetchall()
    conn.close()
    
    unavailable_ranges = []
    for r in booked_rows:
        start, end, total_nights = booking_window_from_row(r)
        if total_nights <= 0:
            continue
        unavailable_ranges.append({
            "start": start.isoformat(),
            "end": end.isoformat()
        })
    
    return render_template("view_room.html", room=room, meal_plans=meal_plans, unavailable_ranges=unavailable_ranges)

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out", "success")
    return redirect(url_for("landing"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        address = request.form["address"]
        password_hash = generate_password_hash(request.form["password"])
        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO customers (name, email, phone, address, password) VALUES (?, ?, ?, ?, ?)",
                (name, email, phone, address, password_hash)
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
        name = request.form["name"].strip()
        email = request.form["email"].strip()
        password = request.form["password"]
        conn = get_db_connection()
        user = conn.execute(
            "SELECT * FROM customers WHERE email=? AND name=?",
            (email, name)
        ).fetchone()
        conn.close()
        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["customer_id"]
            session["is_admin"] = bool(user["is_admin"])
            session["user_name"] = user["name"]
            flash("Login successful!", "success")
            # Check for return URL
            next_url = request.args.get("next") or request.form.get("next")
            if next_url:
                return redirect(next_url)
            return redirect(url_for("admin_dashboard" if user["is_admin"] else "user_dashboard"))
        flash("Invalid credentials", "danger")
    return render_template("login.html", next=request.args.get("next"))

@app.route("/user_dashboard")
@login_required
def user_dashboard():
    if session.get("is_admin"):
        flash("User access required.", "danger")
        return redirect(url_for("admin_dashboard"))
    conn = get_db_connection()
    available_rooms = conn.execute("""
        SELECT t1.*, t2.image_path, t2.room_type_name, t2.description
        FROM rooms t1
        LEFT JOIN room_types t2 ON t1.room_type_id = t2.room_type_id
    """).fetchall() 
    conn.close()
    return render_template("user_dashboard.html", available_rooms=available_rooms)

@app.route("/user_profile", methods=["GET", "POST"])
@login_required
def user_profile():
    user_id = session["user_id"]
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM customers WHERE customer_id = ?", (user_id,)).fetchone()
    if request.method == "POST":
        name = request.form["name"].strip()
        phone = request.form["phone"].strip()
        address = request.form["address"].strip()
        current_password = request.form["current_password"]
        if not check_password_hash(user["password"], current_password):
            flash("Incorrect current password!", "danger")
            return redirect(url_for("user_profile"))
        conn.execute("""
            UPDATE customers SET name = ?, phone = ?, address = ? WHERE customer_id = ?
        """, (name, phone, address, user_id))
        conn.commit()
        flash("Profile updated successfully!", "success")
        return redirect(url_for("user_profile"))
    conn.close()
    return render_template("user_profile.html", user=user)

# ========================================
# BOOK ROOM (MODIFIED FOR KHALTI)
# ========================================
@app.route("/book_room/<int:room_id>", methods=["GET", "POST"])
@login_required
def book_room(room_id):
    conn = get_db_connection()
    room = conn.execute("""
        SELECT r.room_id, r.room_number, r.price_per_night, t.room_type_name, t.image_path, t.description
        FROM rooms r
        JOIN room_types t ON r.room_type_id = t.room_type_id
        WHERE r.room_id=?
    """, (room_id,)).fetchone()
    
    if not room:
        flash("Room not found!", "danger")
        conn.close()
        return redirect(url_for("user_dashboard"))

    meal_plans = conn.execute("SELECT * FROM meal_plans").fetchall()
    segments = conn.execute("SELECT * FROM market_segments").fetchall()
    customer_id = session["user_id"]
    prev_bookings = conn.execute("""
        SELECT booking_status FROM bookings WHERE customer_id = ? ORDER BY created_at DESC
    """, (customer_id,)).fetchall()

    total_previous_bookings = len(prev_bookings)
    previous_cancellations = sum(1 for b in prev_bookings if b["booking_status"] == "Canceled")
    previous_successful = sum(1 for b in prev_bookings if b["booking_status"] == "Not_Canceled")
    repeated_guest = 1 if total_previous_bookings > 0 else 0

    # OFFLINE BOOKING (Pay at reception)
    if request.method == "POST" and request.is_json:
        booking_data = request.get_json()
        try:
            checkin, checkout, total_nights = booking_window_from_payload(booking_data)
            if total_nights <= 0:
                raise ValueError("Stay must be at least 1 night")
            if not is_room_available(booking_data['room_id'], checkin, checkout):
                return jsonify({"success": False, "message": "Room unavailable for selected dates."}), 400

            conn.execute("""
                INSERT INTO bookings (
                    customer_id, room_id, meal_plan_id, market_segment_id, booking_status,
                    no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights,
                    lead_time, arrival_year, arrival_month, arrival_date,
                    avg_price_per_room, no_of_special_requests, required_car_parking_space,
                    repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
                    total_nights, total_guests
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                customer_id, booking_data['room_id'], booking_data['meal_plan_id'], 
                booking_data['market_segment_id'], "Not_Canceled",
                booking_data['no_of_adults'], booking_data['no_of_children'],
                booking_data['no_of_weekend_nights'], booking_data['no_of_week_nights'],
                booking_data['lead_time'], booking_data['arrival_year'], 
                booking_data['arrival_month'], booking_data['arrival_date'],
                booking_data['avg_price_per_room'], booking_data.get('no_of_special_requests', 0),
                booking_data.get('required_car_parking_space', 0),
                booking_data.get('repeated_guest', 0), booking_data.get('no_of_previous_cancellations', 0),
                booking_data.get('no_of_previous_bookings_not_canceled', 0),
                total_nights, booking_data['total_guests']
            ))
            conn.commit()
            conn.close()
            return jsonify({"success": True, "message": f"Room {booking_data['room_number']} booked! Pay at reception."})
        except Exception as e:
            conn.rollback()
            conn.close()
            print(f"Offline booking failed: {e}")
            return jsonify({"success": False, "message": "Booking failed."}), 400

    # Precompute unavailable ranges for display
    booked_rows = conn.execute("""
        SELECT arrival_year, arrival_month, arrival_date,
               total_nights, no_of_weekend_nights, no_of_week_nights
        FROM bookings
        WHERE room_id = ? AND booking_status = 'Not_Canceled'
    """, (room_id,)).fetchall()
    conn.close()

    unavailable_ranges = []
    for r in booked_rows:
        start, end, total_nights = booking_window_from_row(r)
        if total_nights <= 0:
            continue
        unavailable_ranges.append({
            "start": start.isoformat(),
            "end": end.isoformat()
        })

    return render_template(
        "book_room.html", 
        room=room, 
        meal_plans=meal_plans, 
        segments=segments,
        repeated_guest=repeated_guest,
        previous_cancellations=previous_cancellations,
        previous_successful=previous_successful,
        khalti_public_key=KHALTI_PUBLIC_KEY,  # Pass to template
        unavailable_ranges=unavailable_ranges
    )

@app.route("/my_bookings")
@login_required
def my_bookings():
    user_id = session["user_id"]
    conn = get_db_connection()
    bookings = conn.execute("""
        SELECT b.booking_id, r.room_number, t.room_type_name, s.segment_name,
               b.no_of_adults, b.no_of_children, b.total_nights, b.total_guests,
               b.booking_status, b.created_at, b.avg_price_per_room, t.image_path
        FROM bookings b
        LEFT JOIN rooms r ON b.room_id = r.room_id
        LEFT JOIN room_types t ON r.room_type_id = t.room_type_id
        LEFT JOIN market_segments s ON b.market_segment_id = s.market_segment_id
        WHERE b.customer_id = ?
        ORDER BY b.created_at DESC
    """, (user_id,)).fetchall()
    conn.close()
    return render_template("my_bookings.html", bookings=bookings)

@app.route("/cancel_booking/<int:booking_id>", methods=["POST"])
@login_required
def cancel_booking(booking_id):
    user_id = session["user_id"]
    conn = get_db_connection()
    booking = conn.execute(
        "SELECT * FROM bookings WHERE booking_id = ? AND customer_id = ?",
        (booking_id, user_id)
    ).fetchone()
    if not booking:
        flash("Booking not found.", "danger")
        conn.close()
        return redirect(url_for("my_bookings"))
    if booking["booking_status"] == "Canceled":
        flash("Already canceled.", "info")
        conn.close()
        return redirect(url_for("my_bookings"))
    conn.execute(
        "UPDATE bookings SET booking_status = 'Canceled', updated_at = CURRENT_TIMESTAMP WHERE booking_id = ?",
        (booking_id,)
    )
    conn.commit()
    conn.close()
    flash("Booking canceled successfully.", "success")
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
            image_path = os.path.join(app.config["MENU_PLAN_UPLOAD_FOLDER"], image_filename)
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


@app.route("/admin/delete_meal_plan/<int:meal_id>", methods=["GET", "POST"])
@admin_required
def delete_meal_plan(meal_id):
    conn = get_db_connection()

    # Check if meal plan exists
    meal_plan = conn.execute(
        "SELECT meal_plan_id, meal_plan_name, image_path FROM meal_plans WHERE meal_plan_id = ?",
        (meal_id,)
    ).fetchone()

    if not meal_plan:
        flash("Meal plan not found!", "danger")
        conn.close()
        return redirect(url_for("manage_meal_plans"))

    # Check if there are any bookings using this meal plan
    bookings_count = conn.execute(
        "SELECT COUNT(*) FROM bookings WHERE meal_plan_id = ?",
        (meal_id,)
    ).fetchone()[0]

    # If GET request and has bookings, show reassignment page
    if request.method == "GET" and bookings_count > 0:
        # Get all other meal plans for reassignment
        other_meal_plans = conn.execute(
            "SELECT meal_plan_id, meal_plan_name FROM meal_plans WHERE meal_plan_id != ? ORDER BY meal_plan_name",
            (meal_id,)
        ).fetchall()
        conn.close()
        
        if not other_meal_plans:
            flash("Cannot delete: This meal plan is used in bookings and there are no other meal plans to reassign to.", "danger")
            return redirect(url_for("manage_meal_plans"))
        
        return render_template("delete_meal_plan_confirm.html", 
                             meal_plan=meal_plan, 
                             bookings_count=bookings_count,
                             other_meal_plans=other_meal_plans)

    # Handle POST request (actual deletion)
    if request.method == "POST":
        reassign_to = request.form.get("reassign_to")
        
        # If bookings exist, reassign them
        if bookings_count > 0:
            if not reassign_to:
                flash("Please select a meal plan to reassign bookings to.", "danger")
                conn.close()
                return redirect(url_for("delete_meal_plan", meal_id=meal_id))
            
            # Verify reassign_to meal plan exists
            reassign_plan = conn.execute(
                "SELECT meal_plan_id FROM meal_plans WHERE meal_plan_id = ?",
                (reassign_to,)
            ).fetchone()
            
            if not reassign_plan:
                flash("Selected meal plan not found!", "danger")
                conn.close()
                return redirect(url_for("delete_meal_plan", meal_id=meal_id))
            
            # Reassign bookings
            conn.execute(
                "UPDATE bookings SET meal_plan_id = ? WHERE meal_plan_id = ?",
                (reassign_to, meal_id)
            )

        try:
            # Delete image file
            if meal_plan["image_path"]:
                img_path = os.path.join(MENU_PLAN_UPLOAD_FOLDER, meal_plan["image_path"])
                if os.path.exists(img_path):
                    os.remove(img_path)

            # Delete DB row
            conn.execute(
                "DELETE FROM meal_plans WHERE meal_plan_id = ?",
                (meal_id,)
            )

            conn.commit()
            if bookings_count > 0:
                flash(f"Meal plan deleted successfully! {bookings_count} booking(s) reassigned.", "success")
            else:
                flash("Meal plan deleted successfully!", "success")
        except sqlite3.IntegrityError as e:
            conn.rollback()
            flash(f"Cannot delete meal plan: {str(e)}", "danger")
        except Exception as e:
            conn.rollback()
            flash(f"Error deleting meal plan: {str(e)}", "danger")
        finally:
            conn.close()

        return redirect(url_for("manage_meal_plans"))
    
    # If no bookings, proceed with deletion directly
    try:
        # Delete image file
        if meal_plan["image_path"]:
            img_path = os.path.join(MENU_PLAN_UPLOAD_FOLDER, meal_plan["image_path"])
            if os.path.exists(img_path):
                os.remove(img_path)

        # Delete DB row
        conn.execute(
            "DELETE FROM meal_plans WHERE meal_plan_id = ?",
            (meal_id,)
        )

        conn.commit()
        flash("Meal plan deleted successfully!", "success")
    except sqlite3.IntegrityError as e:
        conn.rollback()
        flash(f"Cannot delete meal plan: {str(e)}", "danger")
    except Exception as e:
        conn.rollback()
        flash(f"Error deleting meal plan: {str(e)}", "danger")
    finally:
        conn.close()

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
    # Debug can be toggled via FLASK_DEBUG=true; default is off for safety
    debug_mode = str(os.getenv("FLASK_DEBUG", "")).lower() in ("1", "true", "yes")
    app.run(debug=debug_mode)
