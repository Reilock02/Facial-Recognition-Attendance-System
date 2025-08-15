import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import csv
import os
import pandas as pd
import cv2
from datetime import datetime
from PIL import Image, ImageTk
import ttkbootstrap as bstrap
from ttkbootstrap.toast import ToastNotification
from tkinter import filedialog
import face_recognition
import numpy as np
import pickle
from scipy.spatial import distance as dist
import shutil

class FacialRecognitionAttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Facial Recognition Attendance System")
        self.root.geometry("1250x800")
        
        # --- NEW: Professional and Eye-Friendly Dark Theme ---
        self.setup_custom_theme()

        # --- Configuration and Constants ---
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES_REGISTER = 3
        self.EYE_AR_CONSEC_FRAMES_ATTENDANCE = 2
        self.HEAD_TILT_THRESH = 15
        self.RECOGNITION_COOLDOWN_SECONDS = 5
        self.RECOGNITION_FRAME_INTERVAL = 4

        # --- File and Directory Setup ---
        for folder in ['data', 'icons', 'registered_faces', 'data_backups']:
            if not os.path.exists(folder): os.makedirs(folder)

        self.students_file = 'data/students.csv'
        self.attendance_file = 'data/attendance.csv'
        self.encodings_file = 'data/encodings.pkl'

        self.backup_data_files()
        self.initialize_files()

        # --- State Variables ---
        self.known_face_encodings, self.known_face_ids = [], []
        self.load_known_faces()
        self.cap, self.scanning = None, False
        self.last_recognition_times = {}
        self.frame_counter = 0
        self.last_known_locations, self.last_known_names = [], []
        self.live_blink_counters = {}

        self.create_widgets()
        self.update_scan_status(False) # Set initial status indicator color

    def setup_custom_theme(self):
        """Defines a professional dark theme with cyan accents."""
        self.style = bstrap.Style(theme='darkly')
        
        bg_color = '#222222'
        fg_color = '#DDDDDD'
        accent_color = '#08D9D6' # Cyan
        secondary_color = '#333333'
        success_color = '#28A745'
        danger_color = '#DC3545'
        input_bg = '#3A3A3A'

        # --- Configure widget styles ---
        self.style.configure('.', background=bg_color, foreground=fg_color, font=('Helvetica', 10))
        self.style.configure('TFrame', background=bg_color)
        self.root.configure(background=bg_color)

        self.style.configure('TLabel', background=bg_color, foreground=fg_color)
        self.style.configure('TLabelframe', background=bg_color, bordercolor=secondary_color)
        self.style.configure('TLabelframe.Label', foreground=accent_color, background=bg_color, font=('Helvetica', 12, 'bold'))
        
        self.style.configure('TButton', background=accent_color, foreground=bg_color, font=('Helvetica', 10, 'bold'), borderwidth=0, padding=(10, 5))
        self.style.map('TButton', background=[('active', secondary_color), ('disabled', secondary_color)])
        
        # Specific button styles for clarity
        self.style.configure('danger.TButton', background=danger_color, foreground=fg_color)
        self.style.configure('info-outline.TButton', background=bg_color, foreground=accent_color, borderwidth=1, bordercolor=accent_color)

        self.style.configure('TNotebook', background=bg_color, bordercolor=secondary_color)
        self.style.configure('TNotebook.Tab', background=secondary_color, foreground=fg_color, font=('Helvetica', 11, 'bold'), padding=(10, 5))
        self.style.map('TNotebook.Tab', background=[('selected', bg_color)], foreground=[('selected', accent_color)])

        self.style.configure('Treeview', fieldbackground=input_bg, background=input_bg, foreground=fg_color)
        self.style.configure('Treeview.Heading', background=secondary_color, foreground=fg_color, font=('Helvetica', 10, 'bold'))
        self.style.map('Treeview', background=[('selected', secondary_color)], foreground=[('selected', accent_color)])
        
        self.style.configure('TEntry', fieldbackground=input_bg, foreground=fg_color, insertcolor=fg_color)
        self.style.configure('TCombobox', fieldbackground=input_bg, foreground=fg_color, selectbackground=input_bg, selectforeground=fg_color)
        
        # Styles for the dynamic status label
        self.style.configure('success.TLabel', foreground=success_color, font=('Helvetica', 11, 'bold'))
        self.style.configure('danger.TLabel', foreground=danger_color, font=('Helvetica', 11, 'bold'))

    def backup_data_files(self):
        # This function remains unchanged
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join('data_backups', timestamp)
        try:
            os.makedirs(backup_dir)
            for file in [self.students_file, self.attendance_file, self.encodings_file]:
                if os.path.exists(file): shutil.copy(file, backup_dir)
        except OSError as e: print(f"Error creating backup directory: {e}")

    def initialize_files(self):
        # This function remains unchanged
        if not os.path.exists(self.students_file):
            with open(self.students_file, 'w', newline='') as f: csv.writer(f).writerow(['ID', 'Name'])
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f: csv.writer(f).writerow(['ID', 'Name', 'Date', 'TimeIn', 'TimeOut'])
        if not os.path.exists(self.encodings_file):
            with open(self.encodings_file, 'wb') as f: pickle.dump(([], []), f)

    def load_known_faces(self):
        # This function remains unchanged
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    self.known_face_encodings, self.known_face_ids = pickle.load(f)
            except (EOFError, pickle.UnpicklingError): self.known_face_encodings, self.known_face_ids = [], []

    def save_known_faces(self):
        # This function remains unchanged
        with open(self.encodings_file, 'wb') as f: pickle.dump((self.known_face_encodings, self.known_face_ids), f)

    def load_icon(self, filename, size=(24, 24)):
        # This function remains unchanged
        try:
            path = os.path.join('icons', filename)
            with Image.open(path) as img: return ImageTk.PhotoImage(img.resize(size, Image.LANCZOS))
        except FileNotFoundError: print(f"Warning: Icon file not found at {path}"); return None

    def get_available_cameras(self):
        # This function remains unchanged
        index, arr = 0, []
        while True:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.read()[0]: break
            else: arr.append(index)
            cap.release(); index += 1
        return arr
            
    def create_widgets(self):
        # This function remains unchanged
        self.camera_icon = self.load_icon('camera.png'); self.stop_icon = self.load_icon('stop.png')
        self.add_user_icon = self.load_icon('add-user.png'); self.delete_user_icon = self.load_icon('delete-user.png')
        self.export_icon = self.load_icon('export.png'); self.refresh_icon = self.load_icon('refresh.png')
        self.edit_user_icon = self.load_icon('edit-user.png'); self.history_icon = self.load_icon('history.png')
        main_frame = bstrap.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        notebook = bstrap.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        self.create_attendance_tab(notebook)
        self.create_user_management_tab(notebook)
        self.create_history_tab(notebook)
        self.create_export_tab(notebook)

    def create_attendance_tab(self, notebook):
        # Increased padding for better layout
        attendance_frame = bstrap.Frame(notebook, padding=15)
        notebook.add(attendance_frame, text='   Attendance   ')
        
        controls_frame = bstrap.LabelFrame(attendance_frame, text="Camera Controls", padding=15)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        bstrap.Label(controls_frame, text="Select Camera:").pack(pady=5, anchor='w')
        self.camera_options = self.get_available_cameras()
        self.camera_selection_var = tk.StringVar(value=self.camera_options[0] if self.camera_options else "")
        self.camera_selector = bstrap.Combobox(controls_frame, textvariable=self.camera_selection_var, values=self.camera_options, state="readonly")
        self.camera_selector.pack(pady=(0,15), fill=tk.X)
        if not self.camera_options: self.camera_selector.config(state=tk.DISABLED)

        self.camera_label = bstrap.Label(controls_frame, relief=tk.SOLID, borderwidth=1)
        self.camera_label.pack(pady=10, fill=tk.BOTH, expand=True)

        self.start_scan_button = bstrap.Button(controls_frame, text=" Start Scanning", image=self.camera_icon, compound=tk.LEFT, command=self.start_scanning)
        self.start_scan_button.pack(pady=(15, 5), fill=tk.X)
        if not self.camera_options: self.start_scan_button.config(state=tk.DISABLED)
        
        self.stop_scan_button = bstrap.Button(controls_frame, text=" Stop Scanning", image=self.stop_icon, compound=tk.LEFT, command=self.stop_scanning, bootstyle="danger", state=tk.DISABLED)
        self.stop_scan_button.pack(pady=5, fill=tk.X)
        
        self.scan_status_label = bstrap.Label(controls_frame, text="Status: Not Scanning")
        self.scan_status_label.pack(pady=10, side=tk.BOTTOM)

        log_frame = bstrap.LabelFrame(attendance_frame, text="Real-time Attendance Log", padding=15)
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        refresh_log_button = bstrap.Button(log_frame, text=" Refresh", image=self.refresh_icon, compound=tk.LEFT, command=self.load_attendance, bootstyle="info-outline")
        refresh_log_button.pack(side=tk.RIGHT, pady=(0, 10), anchor='ne')
        
        self.tree = bstrap.Treeview(log_frame, columns=('ID', 'Name', 'Date', 'TimeIn', 'TimeOut'), show='headings')
        for col in self.tree['columns']: self.tree.heading(col, text=col.replace('Time', 'Time '))
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.load_attendance()

    def create_user_management_tab(self, notebook):
        # Reorganized layout for better flow
        user_mgmt_frame = bstrap.Frame(notebook, padding=15)
        notebook.add(user_mgmt_frame, text=' User Management ')

        left_panel = bstrap.Frame(user_mgmt_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        right_panel = bstrap.Frame(user_mgmt_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        reg_frame = bstrap.LabelFrame(left_panel, text="Register New User", padding=15)
        reg_frame.pack(fill=tk.X, pady=(0, 20))
        bstrap.Label(reg_frame, text="User ID:").grid(row=0, column=0, padx=5, pady=10, sticky=tk.W)
        self.user_id_entry = bstrap.Entry(reg_frame, width=40); self.user_id_entry.grid(row=0, column=1, padx=5, pady=10)
        bstrap.Label(reg_frame, text="User Name:").grid(row=1, column=0, padx=5, pady=10, sticky=tk.W)
        self.user_name_entry = bstrap.Entry(reg_frame, width=40); self.user_name_entry.grid(row=1, column=1, padx=5, pady=10)
        register_button = bstrap.Button(reg_frame, text=" Start Interactive Registration", image=self.add_user_icon, compound=tk.LEFT, command=self.register_user)
        register_button.grid(row=2, column=0, columnspan=2, pady=10)

        user_list_frame = bstrap.LabelFrame(left_panel, text="Registered Users", padding=15)
        user_list_frame.pack(fill=tk.BOTH, expand=True)
        self.user_tree = bstrap.Treeview(user_list_frame, columns=('ID', 'Name'), show='headings')
        self.user_tree.heading('ID', text='ID'); self.user_tree.heading('Name', text='Name')
        self.user_tree.pack(fill=tk.BOTH, expand=True)
        self.user_tree.bind('<<TreeviewSelect>>', self.update_user_details_view)
        
        # --- NEW User Details Panel ---
        details_frame = bstrap.LabelFrame(right_panel, text="User Details", padding=15)
        details_frame.pack(fill=tk.Y)
        self.user_photo_label = bstrap.Label(details_frame)
        self.user_photo_label.pack(pady=10, padx=10)
        self.details_id_label = bstrap.Label(details_frame, text="ID: -", font=('Helvetica', 11, 'bold'))
        self.details_id_label.pack(pady=(10,0))
        self.details_name_label = bstrap.Label(details_frame, text="Name: -", font=('Helvetica', 10))
        self.details_name_label.pack(pady=(0,15))
        
        edit_user_button = bstrap.Button(details_frame, text=" Edit User", image=self.edit_user_icon, compound=tk.LEFT, command=self.edit_user)
        edit_user_button.pack(pady=5, fill=tk.X)
        delete_user_button = bstrap.Button(details_frame, text=" Delete User", image=self.delete_user_icon, compound=tk.LEFT, command=self.delete_user, bootstyle="danger")
        delete_user_button.pack(pady=5, fill=tk.X)
        refresh_user_button = bstrap.Button(details_frame, text=" Refresh List", image=self.refresh_icon, compound=tk.LEFT, command=self.load_users, bootstyle="info-outline")
        refresh_user_button.pack(pady=(15, 5), fill=tk.X)

        self.load_users()

    def create_history_tab(self, notebook):
        # This function remains largely unchanged, just styled
        history_frame = bstrap.Frame(notebook, padding=15)
        notebook.add(history_frame, text=' Attendance History ')
        filter_frame = bstrap.LabelFrame(history_frame, text="Filter by Month", padding=15)
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        bstrap.Label(filter_frame, text="Year:").pack(side=tk.LEFT, padx=(0, 5))
        self.year_combo = bstrap.Combobox(filter_frame, state="readonly", width=10)
        self.year_combo.pack(side=tk.LEFT, padx=5)
        bstrap.Label(filter_frame, text="Month:").pack(side=tk.LEFT, padx=(15, 5))
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        self.month_combo = bstrap.Combobox(filter_frame, state="readonly", values=months, width=15)
        self.month_combo.pack(side=tk.LEFT, padx=5)
        self.month_combo.set(datetime.now().strftime("%B"))
        load_button = bstrap.Button(filter_frame, text="Load Data", command=self.load_historical_data)
        load_button.pack(side=tk.LEFT, padx=20)
        export_history_button = bstrap.Button(filter_frame, text="Export this View", image=self.export_icon, compound=tk.LEFT, command=self.export_history_view, bootstyle="info-outline")
        export_history_button.pack(side=tk.RIGHT, padx=5)
        self.populate_year_filter()
        history_data_frame = bstrap.Frame(history_frame)
        history_data_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.history_tree = bstrap.Treeview(history_data_frame, columns=('ID', 'Name', 'Date', 'TimeIn', 'TimeOut'), show='headings')
        for col in self.history_tree['columns']: self.history_tree.heading(col, text=col.replace('Time', 'Time '))
        self.history_tree.pack(fill=tk.BOTH, expand=True)

    def create_export_tab(self, notebook):
        # This function remains largely unchanged, just styled
        export_frame = bstrap.Frame(notebook, padding=20)
        notebook.add(export_frame, text='   Export Full Log   ')
        export_info_label = bstrap.Label(export_frame, text="Export the ENTIRE attendance log to a single CSV file. To export a specific month, use the 'Export this View' button in the 'Attendance History' tab.", wraplength=500, justify=tk.CENTER)
        export_info_label.pack(pady=20)
        export_button = bstrap.Button(export_frame, text=" Export Full Attendance to CSV", image=self.export_icon, compound=tk.LEFT, command=self.export_full_attendance)
        export_button.pack(pady=20)

    # --- NEW AND IMPROVED FUNCTIONS ---
    def update_scan_status(self, is_scanning):
        """Updates the status label text and color."""
        if is_scanning:
            self.scan_status_label.config(text="Status: Scanning...", style="success.TLabel")
        else:
            self.scan_status_label.config(text="Status: Not Scanning", style="danger.TLabel")

    def update_user_details_view(self, event=None):
        """NEW: Displays selected user's photo and details in the side panel."""
        selected_items = self.user_tree.selection()
        if not selected_items:
            self.user_photo_label.config(image=''); self.user_photo_label.image = None
            self.details_id_label.config(text="ID: -")
            self.details_name_label.config(text="Name: -")
            return

        user_id, user_name = self.user_tree.item(selected_items[0])['values']
        self.details_id_label.config(text=f"ID: {user_id}")
        self.details_name_label.config(text=f"Name: {user_name}")
        
        photo_path = os.path.join('registered_faces', f"{user_id}.jpg")
        if os.path.exists(photo_path):
            try:
                img = Image.open(photo_path)
                img.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(img)
                self.user_photo_label.config(image=photo)
                self.user_photo_label.image = photo # Keep a reference
            except Exception as e:
                print(f"Error loading image: {e}"); self.user_photo_label.config(image='')
        else:
            self.user_photo_label.config(image=''); self.user_photo_label.image = None

    def register_user(self):
        # IMPROVED: Better UI for the registration window
        user_id, user_name = self.user_id_entry.get().strip(), self.user_name_entry.get().strip()
        if not user_id or not user_name: self.show_toast("Registration Error", "User ID and Name cannot be empty.", "danger"); return
        try:
            if int(user_id) in pd.read_csv(self.students_file)['ID'].values: self.show_toast("Registration Error", "This User ID already exists.", "danger"); return
        except (ValueError, FileNotFoundError, pd.errors.EmptyDataError): pass # Handles multiple potential issues gracefully
        
        cap = cv2.VideoCapture(int(self.camera_selection_var.get()));
        if not cap.isOpened(): self.show_toast("Camera Error", "Could not open webcam.", "danger"); return
        
        step, blink_counter, fwd_frame, instruction = "LOOK_STRAIGHT", 0, None, "Position face in oval and Press Spacebar"
        accent_color_bgr = (214, 217, 8) # BGR format for OpenCV

        while True:
            ret, frame = cap.read();
            if not ret: break
            h, w, _ = frame.shape
            
            # --- New Registration UI Elements ---
            # Guideline Oval
            center_x, center_y = w // 2, h // 2
            cv2.ellipse(frame, (center_x, center_y), (w//4, h//3), 0, 0, 360, accent_color_bgr, 2)
            # Translucent instruction bar
            overlay = frame.copy(); cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            # Centered Text
            (text_w, text_h), _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(frame, instruction, (center_x - text_w // 2, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, accent_color_bgr, 2)

            rgb_frame, face_locs = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), face_recognition.face_locations(rgb_frame)
            if len(face_locs) == 1:
                landmarks = face_recognition.face_landmarks(rgb_frame, face_locs)[0]
                if step == "LOOK_STRAIGHT" and cv2.waitKey(1) & 0xFF == 32: fwd_frame, step, instruction = frame.copy(), "BLINK", "Great! Now blink three times."
                elif step == "BLINK":
                    ear = (self.eye_aspect_ratio(landmarks['left_eye']) + self.eye_aspect_ratio(landmarks['right_eye'])) / 2.0
                    if ear < self.EYE_AR_THRESH: blink_counter += 1
                    else:
                        if blink_counter >= self.EYE_AR_CONSEC_FRAMES_REGISTER: step, instruction = "TILT_LEFT", f"Tilt head left > {self.HEAD_TILT_THRESH} deg"
                        blink_counter = 0
                elif "TILT" in step:
                    angle = np.degrees(np.arctan2(landmarks['chin'][8][1] - landmarks['nose_tip'][0][1], landmarks['chin'][8][0] - landmarks['nose_tip'][0][0])) - 90
                    if step == "TILT_LEFT" and angle > self.HEAD_TILT_THRESH: step, instruction = "TILT_RIGHT", f"Perfect! Now tilt head right > {self.HEAD_TILT_THRESH} deg"
                    elif step == "TILT_RIGHT" and angle < -self.HEAD_TILT_THRESH: step, instruction = "DONE", "All set! Registering..."
            if step == "DONE":
                encs = face_recognition.face_encodings(cv2.cvtColor(fwd_frame, cv2.COLOR_BGR2RGB))
                if encs:
                    self.known_face_encodings.append(encs[0]); self.known_face_ids.append(user_id); self.save_known_faces()
                    with open(self.students_file, 'a', newline='') as f: csv.writer(f).writerow([user_id, user_name])
                    cv2.imwrite(os.path.join('registered_faces', f"{user_id}.jpg"), fwd_frame)
                    self.show_toast("Success", f"User {user_name} registered."); self.user_id_entry.delete(0, tk.END); self.user_name_entry.delete(0, tk.END); self.load_users(); break
            cv2.imshow("Interactive Registration", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
        cap.release(); cv2.destroyAllWindows()

    # --- Unchanged Core Logic Functions ---
    def start_scanning(self):
        try: camera_index = int(self.camera_selection_var.get()); self.cap = cv2.VideoCapture(camera_index)
        except (ValueError, IndexError): self.show_toast("Camera Error", "Invalid camera.", "danger"); return
        if not self.cap.isOpened(): self.show_toast("Camera Error", f"Could not open camera {camera_index}.", "danger"); return
        self.scanning, self.frame_counter = True, 0; self.live_blink_counters.clear(); self.scan_loop()
        self.start_scan_button.config(state=tk.DISABLED); self.stop_scan_button.config(state=tk.NORMAL)
        self.update_scan_status(True)

    def stop_scanning(self):
        self.scanning = False
        if self.cap: self.cap.release(); self.cap = None
        self.start_scan_button.config(state=tk.NORMAL); self.stop_scan_button.config(state=tk.DISABLED)
        self.camera_label.config(image=''); self.camera_label.configure(text="")
        self.update_scan_status(False)

    def scan_loop(self):
        # This function's logic is unchanged
        if not self.scanning or not self.cap or not self.cap.isOpened(): self.stop_scanning(); return
        ret, frame = self.cap.read();
        if not ret: self.stop_scanning(); return
        if self.frame_counter % self.RECOGNITION_FRAME_INTERVAL == 0:
            small_frame, rgb_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25), cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_BGR2RGB)
            self.last_known_locations, face_encs = face_recognition.face_locations(rgb_small), face_recognition.face_encodings(rgb_small, self.last_known_locations)
            landmarks = face_recognition.face_landmarks(rgb_small, self.last_known_locations)
            students_df = pd.read_csv(self.students_file) if os.path.exists(self.students_file) else pd.DataFrame(columns=['ID', 'Name'])
            self.last_known_names = []
            for i, enc in enumerate(face_encs):
                matches, name, face_id = face_recognition.compare_faces(self.known_face_encodings, enc, tolerance=0.5), "Unknown", None
                if any(matches):
                    best_match_idx = np.argmin(face_recognition.face_distance(self.known_face_encodings, enc))
                    if matches[best_match_idx]:
                        face_id = self.known_face_ids[best_match_idx]
                        ear = (self.eye_aspect_ratio(landmarks[i]['left_eye']) + self.eye_aspect_ratio(landmarks[i]['right_eye'])) / 2.0
                        if ear < self.EYE_AR_THRESH: self.live_blink_counters[face_id] = self.live_blink_counters.get(face_id, 0) + 1
                        else:
                            if self.live_blink_counters.get(face_id, 0) >= self.EYE_AR_CONSEC_FRAMES_ATTENDANCE: self.log_attendance(face_id)
                            self.live_blink_counters[face_id] = 0
                        try:
                            user_info = students_df[students_df['ID'] == int(face_id)]
                            if not user_info.empty: name = user_info.iloc[0]['Name']
                        except ValueError: pass
                self.last_known_names.append({"name": name, "id": face_id})
        self.frame_counter += 1
        for i, (top, right, bottom, left) in enumerate(self.last_known_locations):
            top *= 4; right *= 4; bottom *= 4; left *= 4; name, face_id = self.last_known_names[i]["name"], self.last_known_names[i]["id"]
            color = (0, 0, 255)
            if name != "Unknown": color = (0, 255, 0) if self.live_blink_counters.get(face_id, 0) > 0 else (0, 255, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2); cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.imgtk = imgtk; self.camera_label.config(image=imgtk); self.root.after(10, self.scan_loop)
    
    # --- All remaining functions are unchanged ---
    def populate_year_filter(self):
        current_year = datetime.now().year; years = {current_year}
        try:
            if os.path.exists(self.attendance_file):
                df = pd.read_csv(self.attendance_file)
                if not df.empty:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df.dropna(subset=['Date'], inplace=True)
                    years.update(df['Date'].dt.year.unique())
        except Exception as e: print(f"Could not populate year filter: {e}")
        sorted_years = sorted(list(years), reverse=True); self.year_combo['values'] = sorted_years; self.year_combo.set(current_year)
    def load_historical_data(self):
        for i in self.history_tree.get_children(): self.history_tree.delete(i)
        year, month_name = self.year_combo.get(), self.month_combo.get()
        if not year or not month_name: self.show_toast("Filter Error", "Please select both a year and a month.", "warning"); return
        try:
            month_num = datetime.strptime(month_name, "%B").month
            df = pd.read_csv(self.attendance_file); df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            filtered_df = df[(df['Date'].dt.year == int(year)) & (df['Date'].dt.month == month_num)]
            if filtered_df.empty: self.show_toast("No Data", f"No records found for {month_name} {year}.", "info"); return
            filtered_df['Date'] = filtered_df['Date'].dt.strftime('%Y-%m-%d')
            for _, row in filtered_df.iloc[::-1].iterrows():
                self.history_tree.insert("", tk.END, values=(row['ID'], row['Name'], row['Date'], row['TimeIn'], row['TimeOut']))
        except (FileNotFoundError, pd.errors.EmptyDataError): self.show_toast("No Data", "Attendance file is empty or missing.", "info")
        except Exception as e: self.show_toast("Error", f"Could not load data: {e}", "danger")
    def export_history_view(self):
        year, month_name = self.year_combo.get(), self.month_combo.get()
        if not year or not month_name: self.show_toast("Export Error", "Please select a year and month first.", "warning"); return
        try:
            month_num = datetime.strptime(month_name, "%B").month
            df = pd.read_csv(self.attendance_file); df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            filtered_df = df[(df['Date'].dt.year == int(year)) & (df['Date'].dt.month == month_num)]
            if filtered_df.empty: self.show_toast("Export Error", "No data in the current view to export.", "warning"); return
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title=f"Save {month_name} {year} Log", initialfile=f"attendance_log_{year}-{month_num:02d}.csv")
            if save_path: filtered_df.to_csv(save_path, index=False); self.show_toast("Export Successful", f"Data saved to {os.path.basename(save_path)}")
        except (FileNotFoundError, pd.errors.EmptyDataError): self.show_toast("Export Error", "Attendance file is empty or missing.", "warning")
        except Exception as e: self.show_toast("Export Error", f"Failed to export data: {e}", "danger")
    def show_toast(self, title, message, bootstyle="success"):
        toast = ToastNotification(title=title, message=message, duration=3000, bootstyle=bootstyle, position=(20, 20, 'se'))
        toast.show_toast()
    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5]); B = dist.euclidean(eye[2], eye[4]); C = dist.euclidean(eye[0], eye[3]); return (A + B) / (2.0 * C)
    def edit_user(self):
        selected_item = self.user_tree.selection()
        if not selected_item: self.show_toast("Edit Error", "Please select a user to edit.", "warning"); return
        user_id, old_name = self.user_tree.item(selected_item, 'values')
        new_name = simpledialog.askstring("Edit User Name", f"Enter new name for User ID {user_id}:", initialvalue=old_name)
        if new_name and new_name.strip() and new_name != old_name:
            new_name = new_name.strip()
            try:
                s_df = pd.read_csv(self.students_file); s_df.loc[s_df['ID'] == int(user_id), 'Name'] = new_name; s_df.to_csv(self.students_file, index=False)
                if os.path.exists(self.attendance_file):
                    a_df = pd.read_csv(self.attendance_file)
                    if not a_df.empty: a_df.loc[a_df['ID'] == int(user_id), 'Name'] = new_name; a_df.to_csv(self.attendance_file, index=False)
                self.show_toast("Success", f"Updated {old_name} to {new_name}."); self.load_users(); self.load_attendance()
            except Exception as e: self.show_toast("Edit Error", f"An error occurred: {e}", "danger")
    def delete_user(self):
        selected_item = self.user_tree.selection()
        if not selected_item: self.show_toast("Deletion Error", "Please select a user.", "warning"); return
        user_id_str, user_name = self.user_tree.item(selected_item, 'values')
        if not messagebox.askyesno("Confirm Deletion", f"Delete {user_name} (ID: {user_id_str})? This is permanent."): return
        try:
            user_id_int = int(user_id_str)
            if os.path.exists(self.students_file): df = pd.read_csv(self.students_file); df[df['ID'] != user_id_int].to_csv(self.students_file, index=False)
            if os.path.exists(self.attendance_file): df = pd.read_csv(self.attendance_file); df[df['ID'] != user_id_int].to_csv(self.attendance_file, index=False)
            indices = [i for i, fid in enumerate(self.known_face_ids) if fid != user_id_str]
            self.known_face_encodings, self.known_face_ids = [self.known_face_encodings[i] for i in indices], [self.known_face_ids[i] for i in indices]
            self.save_known_faces(); img_path = os.path.join('registered_faces', f"{user_id_str}.jpg")
            if os.path.exists(img_path): os.remove(img_path)
            self.show_toast("Success", f"User {user_name} removed."); self.load_users();
        except Exception as e: self.show_toast("Deletion Error", f"An error occurred: {e}", "danger")
    def log_attendance(self, face_id):
        now = datetime.now()
        if (now - self.last_recognition_times.get(face_id, datetime.min)).total_seconds() < self.RECOGNITION_COOLDOWN_SECONDS: return
        try:
            s_df = pd.read_csv(self.students_file); user_info = s_df[s_df['ID'] == int(face_id)]
            if user_info.empty: return
            user_name, date, time = user_info.iloc[0]['Name'], now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')
            try: a_df = pd.read_csv(self.attendance_file)
            except (FileNotFoundError, pd.errors.EmptyDataError): a_df = pd.DataFrame(columns=['ID', 'Name', 'Date', 'TimeIn', 'TimeOut'])
            today_rec = a_df[(a_df['ID'] == int(face_id)) & (a_df['Date'] == date)]
            if today_rec.empty:
                new_entry = pd.DataFrame([{'ID': int(face_id), 'Name': user_name, 'Date': date, 'TimeIn': time, 'TimeOut': '---'}])
                a_df = pd.concat([a_df, new_entry], ignore_index=True); self.show_toast("Time In", f"{user_name} clocked in.")
            else: a_df.loc[today_rec.index[0], 'TimeOut'] = time; self.show_toast("Time Out", f"{user_name} clocked out.", "info")
            a_df.to_csv(self.attendance_file, index=False); self.load_attendance(); self.last_recognition_times[face_id] = now
        except Exception as e: self.show_toast("Logging Error", f"Error: {e}", "danger")
    def load_attendance(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        try:
            df = pd.read_csv(self.attendance_file)
            if not df.empty:
                for _, row in df.iloc[::-1].iterrows(): self.tree.insert("", tk.END, values=(row['ID'], row['Name'], row['Date'], row['TimeIn'], row['TimeOut']))
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e: print(f"Could not load attendance: {e}")
    def load_users(self):
        for i in self.user_tree.get_children(): self.user_tree.delete(i)
        try:
            df = pd.read_csv(self.students_file)
            if not df.empty:
                for _, row in df.iterrows(): self.user_tree.insert("", tk.END, values=(row['ID'], row['Name']))
        except (FileNotFoundError, pd.errors.EmptyDataError): pass
        self.update_user_details_view() # Clear details panel
    def export_full_attendance(self):
        try:
            if not os.path.exists(self.attendance_file) or pd.read_csv(self.attendance_file).empty:
                self.show_toast("Export Error", "No attendance data to export.", "warning"); return
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save Full Attendance Log As", initialfile=f"full_attendance_log_{datetime.now().strftime('%Y-%m-%d')}.csv")
            if save_path: shutil.copy(self.attendance_file, save_path); self.show_toast("Export Successful", f"Data saved to {os.path.basename(save_path)}")
        except Exception as e: self.show_toast("Export Error", f"Failed to export data: {e}", "danger")

if __name__ == "__main__":
    root = bstrap.Window() 
    app = FacialRecognitionAttendanceSystem(root)
    def on_closing():
        if app.scanning: app.stop_scanning()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()