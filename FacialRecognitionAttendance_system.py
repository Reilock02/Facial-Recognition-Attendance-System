import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import csv
import os
import pandas as pd
import cv2
# --- FIX: Renamed the imported 'time' class to 'dt_time' to avoid conflict with the 'time' module ---
from datetime import datetime, time as dt_time, timedelta
from PIL import Image, ImageTk
import ttkbootstrap as bstrap
from ttkbootstrap.toast import ToastNotification
from ttkbootstrap.widgets import DateEntry
import face_recognition
import numpy as np
import pickle
from scipy.spatial import distance as dist
import shutil
import threading
import configparser
import logging
import sys
# --- FIX: This is the standard 'time' module, which will no longer be overwritten ---
import time
import re

try:
    import winsound
except ImportError:
    winsound = None

class FacialRecognitionAttendanceSystem:
    """
    An advanced facial recognition attendance system with a graphical user interface
    built using Tkinter and ttkbootstrap. It includes features for user registration
    with liveness detection, real-time attendance logging, user management,
    and data export with enhanced visual and audio notifications.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Facial Recognition Attendance System")
        
        try:
            self.root.iconbitmap('icons/app_icon.ico')
        except tk.TclError:
            print("Warning: 'app_icon.ico' not found for the window. Using default icon.")

        self.root.geometry("1250x800")
        
        for folder in ['data', 'icons', 'registered_faces', 'data_backups', 'sounds']:
            os.makedirs(folder, exist_ok=True)

        self.setup_logging()

        self.config = configparser.ConfigParser()
        self.config_file = 'config.ini'
        self.load_config()

        self.setup_custom_theme()
        
        self.setup_sound()

        self.students_file = 'data/students.csv'
        self.attendance_file = 'data/attendance.csv'
        self.scan_log_file = 'data/scan_log.csv'
        self.encodings_file = 'data/encodings.pkl'

        self.backup_data_files()
        self.initialize_files()

        self.known_face_encodings, self.known_face_ids = [], []
        self.cap, self.scanning = None, False
        self.last_recognition_times = {}
        self.last_known_locations, self.last_known_names = [], []
        self.live_blink_counters = {}
        self.editing_user_id = None
        
        self.processing_thread = None
        self.camera_thread = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        self.PROCESS_EVERY_N_FRAMES = 5
        self.frame_counter = 0

        self.create_widgets()
        self.update_scan_status(False)

        self.show_loading_screen()
        self.root.after(100, self.load_initial_data)

    def setup_logging(self):
        """Configures logging to save errors to a file."""
        log_file = os.path.join('data', 'error_log.txt')
        logging.basicConfig(
            filename=log_file,
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        class StreamToLogger:
            def __init__(self, logger, level):
                self.logger = logger
                self.level = level
                self.linebuf = ''
            def write(self, buf):
                for line in buf.rstrip().splitlines():
                    self.logger.log(self.level, line.rstrip())
            def flush(self):
                pass
        sys.stderr = StreamToLogger(logging.getLogger(), logging.ERROR)

    def load_config(self):
        """Loads settings from config.ini or creates it with defaults."""
        if not os.path.exists(self.config_file):
            self.config['Settings'] = {
                'RecognitionCooldownSeconds': '60',
                'EyeAspectRatioThreshold': '0.25',
                'HeadTiltThreshold': '15'
            }
            with open(self.config_file, 'w') as configfile:
                self.config.write(configfile)
        
        self.config.read(self.config_file)
        settings = self.config['Settings']
        self.RECOGNITION_COOLDOWN_SECONDS = settings.getint('RecognitionCooldownSeconds', 60)
        self.EYE_AR_THRESH = settings.getfloat('EyeAspectRatioThreshold', 0.25)
        self.HEAD_TILT_THRESH = settings.getint('HeadTiltThreshold', 15)
        self.EYE_AR_CONSEC_FRAMES_REGISTER = 3
        self.EYE_AR_CONSEC_FRAMES_ATTENDANCE = 2

    def save_config(self):
        """Saves the current settings to the config.ini file."""
        try:
            with open(self.config_file, 'w') as configfile:
                self.config.write(configfile)
            self.show_toast("Settings Saved", "Changes will apply on next launch.", "success")
        except Exception as e:
            self.show_toast("Error", f"Could not save settings: {e}", "danger")
            logging.error(f"Failed to save config: {e}")
    
    def setup_sound(self):
        """Initializes the sound notification system and checks for sound files."""
        self.sound_enabled = False
        print("--- Initializing Sound System ---")

        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()

        sounds_folder = os.path.join(script_dir, 'sounds')

        self.sound_files = {
            'in': os.path.join(sounds_folder, 'time_in.wav'),
            'out': os.path.join(sounds_folder, 'time_out.wav'),
            'late': os.path.join(sounds_folder, 'late.wav')
        }

        if winsound and sys.platform == "win32":
            self.sound_enabled = True
            print("Sound notifications enabled (Windows).")
            for event, path in self.sound_files.items():
                if not os.path.exists(path):
                    print(f"  [WARNING] Sound file for '{event}' NOT FOUND at: {path}. This sound will not play.")
        else:
            print("Warning: Sound notifications are only supported on Windows. Sound is disabled.")


    def play_sound(self, sound_type):
        """Plays a specific sound based on the event type in a separate thread."""
        if not self.sound_enabled:
            return

        sound_path = self.sound_files.get(sound_type)
        if sound_path and os.path.exists(sound_path):
            try:
                threading.Thread(
                    target=lambda: winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC),
                    daemon=True
                ).start()
            except Exception as e:
                logging.error(f"Failed to play sound {sound_path}: {e}")
        else:
            try:
                threading.Thread(
                    target=lambda: winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC),
                    daemon=True
                ).start()
            except Exception as e:
                logging.error(f"Failed to play fallback sound: {e}")

    def setup_custom_theme(self):
        """Defines a professional theme for the application."""
        self.style = bstrap.Style.get_instance()
        
        bg_color = '#FFFFFF'
        widget_bg = '#F9FAFB'
        text_primary = '#111827'
        text_secondary = '#6B7280'
        border_color = '#E5E7EB'
        primary_color = '#800000'
        accent_color = '#FFD700'
        success_color = '#16A34A'
        danger_color = '#DC2626'

        self.root.configure(background=bg_color)
        self.style.configure('.', background=bg_color, foreground=text_primary, font=('Inter', 10))
        self.style.configure('TFrame', background=bg_color)
        self.style.configure('TLabel', background=bg_color, foreground=text_primary)
        self.style.configure('TLabelframe', background=widget_bg, bordercolor=border_color, relief='solid', borderwidth=1)
        self.style.configure('TLabelframe.Label', foreground=primary_color, background=widget_bg, font=('Inter', 12, 'bold'))
        self.style.configure('TButton', background=accent_color, foreground='#42200A', font=('Inter', 10, 'bold'), borderwidth=0, padding=(12, 6))
        self.style.map('TButton', background=[('active', '#FBBF24'), ('disabled', '#D1D5DB')])
        self.style.configure('danger.TButton', background=danger_color, foreground='#FFFFFF')
        self.style.configure('info.TButton', background='#2563EB', foreground='#FFFFFF')
        self.style.configure('info-outline.TButton', background=widget_bg, foreground=primary_color, borderwidth=1, bordercolor=primary_color)
        self.style.map('info-outline.TButton', background=[('active', '#FEF2F2')], bordercolor=[('active', '#F87171')])
        self.style.configure('TNotebook.Tab', background='#F3F4F6', foreground=text_secondary, font=('Inter', 11, 'bold'), padding=(12, 8), borderwidth=0)
        self.style.map('TNotebook.Tab', background=[('selected', bg_color)], foreground=[('selected', primary_color)])
        self.style.configure('Treeview', fieldbackground=widget_bg, background=widget_bg, foreground=text_primary, rowheight=30)
        self.style.configure('Treeview.Heading', background=bg_color, foreground=text_secondary, font=('Inter', 9, 'bold'), padding=8, relief='flat')
        self.style.map('Treeview', background=[('selected', '#FEF2F2')], foreground=[('selected', primary_color)])
        self.style.configure('TEntry', fieldbackground=bg_color, foreground=text_primary, insertcolor=text_primary, bordercolor=border_color, padding=6)
        self.style.map('TEntry', bordercolor=[('focus', primary_color)])
        self.style.configure('success.TLabel', foreground=success_color, font=('Inter', 11, 'bold'))
        self.style.configure('danger.TLabel', foreground=danger_color, font=('Inter', 11, 'bold'))

    def show_loading_screen(self):
        """Displays a simple loading screen."""
        self.loading_frame = bstrap.Frame(self.root, style='TFrame')
        self.loading_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        bstrap.Label(self.loading_frame, text="Loading System Data...", font=('Inter', 14, 'bold')).pack(pady=10)
        self.progress_bar = bstrap.Progressbar(self.loading_frame, mode='indeterminate')
        self.progress_bar.pack(pady=10, fill=tk.X, padx=20)
        self.progress_bar.start()

    def hide_loading_screen(self):
        """Hides the loading screen."""
        self.progress_bar.stop()
        self.loading_frame.destroy()

    def load_initial_data(self):
        """Loads all necessary data at startup and then auto-starts the scanner."""
        try:
            self.load_known_faces()
            self.load_attendance()
            self.load_users()
            self.root.after(500, self.auto_start_scanning)
        except Exception as e:
            logging.error(f"Error during initial data load: {e}")
            messagebox.showerror("Startup Error", "Failed to load initial data. Check error_log.txt for details.")
        finally:
            self.hide_loading_screen()
            self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

    def auto_start_scanning(self):
        """Automatically starts the scanning process if a camera is available."""
        if self.camera_options:
            print("Camera found. Auto-starting scanner...")
            self.start_scanning()
        else:
            print("No camera found. Scanner will not auto-start.")

    def backup_data_files(self):
        """Creates a timestamped backup of critical data files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join('data_backups', timestamp)
        try:
            os.makedirs(backup_dir, exist_ok=True)
            for file in [self.students_file, self.attendance_file, self.encodings_file, self.scan_log_file]:
                if os.path.exists(file):
                    shutil.copy(file, backup_dir)
        except OSError as e:
            logging.error(f"Error creating backup directory: {e}")

    def initialize_files(self):
        """Ensures that necessary CSV and pickle files exist."""
        if not os.path.exists(self.students_file):
            with open(self.students_file, 'w', newline='') as f:
                csv.writer(f).writerow(['ID', 'Name', 'ScheduleDays', 'ScheduleTimeIn', 'ScheduleTimeOut'])
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                csv.writer(f).writerow(['ID', 'Name', 'Date', 'TimeIn', 'TimeOut'])
        if not os.path.exists(self.scan_log_file):
            with open(self.scan_log_file, 'w', newline='') as f:
                csv.writer(f).writerow(['ID', 'Name', 'Date', 'Time'])
        if not os.path.exists(self.encodings_file):
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(([], []), f)

    def load_known_faces(self):
        """Loads face encodings and corresponding IDs from a pickle file."""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    self.known_face_encodings, self.known_face_ids = pickle.load(f)
            except Exception as e:
                logging.error(f"Failed to load encodings file: {e}")
                self.known_face_encodings, self.known_face_ids = [], []

    def save_known_faces(self):
        """Saves the current face encodings and IDs to a pickle file."""
        try:
            with open(self.encodings_file, 'wb') as f:
                pickle.dump((self.known_face_encodings, self.known_face_ids), f)
        except Exception as e:
            logging.error(f"Failed to save encodings file: {e}")
            self.show_toast("Error", "Could not save face data.", "danger")

    def load_icon(self, filename, size=(24, 24)):
        """Loads an icon from the 'icons' folder."""
        try:
            path = os.path.join('icons', filename)
            with Image.open(path) as img:
                return ImageTk.PhotoImage(img.resize(size, Image.LANCZOS))
        except Exception as e:
            logging.error(f"Icon not found at {path}: {e}")
            return None

    def get_available_cameras(self):
        """Detects and returns a list of available camera indices using a more stable backend."""
        index, arr = 0, []
        while index < 10: # Check first 10 indices
            cap = cv2.VideoCapture(index + cv2.CAP_DSHOW)
            if cap.isOpened():
                arr.append(index)
                cap.release()
            index += 1
        print(f"Available cameras found at indices: {arr}" if arr else "No valid cameras found.")
        return arr
            
    def create_widgets(self):
        """Creates and arranges all the GUI widgets."""
        self.camera_icon = self.load_icon('camera.png')
        self.stop_icon = self.load_icon('stop.png')
        self.add_user_icon = self.load_icon('add-user.png')
        self.delete_user_icon = self.load_icon('delete-user.png')
        self.export_icon = self.load_icon('export.png')
        self.refresh_icon = self.load_icon('refresh.png')
        self.edit_user_icon = self.load_icon('edit-user.png')
        self.save_icon = self.load_icon('save.png')
        self.cancel_icon = self.load_icon('cancel.png')
        
        self.toast_icons = {
            'success': self.load_icon('success.png', size=(32, 32)),
            'info': self.load_icon('info.png', size=(32, 32)),
            'warning': self.load_icon('warning.png', size=(32, 32)),
            'danger': self.load_icon('danger.png', size=(32, 32))
        }
        
        main_frame = bstrap.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.notebook = bstrap.Notebook(main_frame)
        
        self.create_attendance_tab(self.notebook)
        self.create_user_management_tab(self.notebook)
        self.create_history_tab(self.notebook)
        self.create_export_tab(self.notebook)
        self.create_settings_tab(self.notebook)

    def create_attendance_tab(self, notebook):
        """Creates the main 'Attendance' tab with camera controls and live log."""
        attendance_frame = bstrap.Frame(notebook, padding=15)
        notebook.add(attendance_frame, text='   Attendance   ')
        
        controls_frame = bstrap.LabelFrame(attendance_frame, text="Camera Controls", padding=15)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        bstrap.Label(controls_frame, text="Select Camera:").pack(pady=5, anchor='w')
        self.camera_options = self.get_available_cameras()
        self.camera_selection_var = tk.StringVar(value=self.camera_options[0] if self.camera_options else "")
        self.camera_selector = bstrap.Combobox(controls_frame, textvariable=self.camera_selection_var, values=self.camera_options, state="readonly")
        self.camera_selector.pack(pady=(0,15), fill=tk.X)
        
        self.camera_label = bstrap.Label(controls_frame, relief=tk.SOLID, borderwidth=1, background="#E5E7EB")
        self.camera_label.pack(pady=10, fill=tk.BOTH, expand=True)

        self.start_scan_button = bstrap.Button(controls_frame, text=" Start Scanning", image=self.camera_icon, compound=tk.LEFT, command=self.start_scanning)
        self.start_scan_button.pack(pady=(15, 5), fill=tk.X)
        
        self.stop_scan_button = bstrap.Button(controls_frame, text=" Stop Scanning", image=self.stop_icon, compound=tk.LEFT, command=self.stop_scanning, bootstyle="danger", state=tk.DISABLED)
        self.stop_scan_button.pack(pady=5, fill=tk.X)
        
        if not self.camera_options:
            self.camera_selector.config(state=tk.DISABLED)
            self.start_scan_button.config(state=tk.DISABLED)
            self.camera_label.config(text="\n\nNo Camera Found", font=('Inter', 12))

        self.scan_status_label = bstrap.Label(controls_frame, text="Status: Not Scanning")
        self.scan_status_label.pack(pady=10, side=tk.BOTTOM)

        log_frame = bstrap.LabelFrame(attendance_frame, text="Real-time Attendance Log (First & Last Scan)", padding=15)
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        refresh_log_button = bstrap.Button(log_frame, text=" Refresh", image=self.refresh_icon, compound=tk.LEFT, command=self.load_attendance, bootstyle="info-outline")
        refresh_log_button.pack(side=tk.RIGHT, pady=(0, 10), anchor='ne')
        
        self.tree = bstrap.Treeview(log_frame, columns=('ID', 'Name', 'Date', 'TimeIn', 'TimeOut'), show='headings')
        
        for col in self.tree['columns']:
            self.tree.heading(col, text=col.replace('Time', 'Time '))
            self.tree.column(col, anchor='center', width=120)
        self.tree.column('ID', width=50)
        self.tree.column('Name', anchor='w', width=150)
                
        self.tree.pack(fill=tk.BOTH, expand=True)

    def create_user_management_tab(self, notebook):
        """Creates the 'User Management' tab for registering and editing users."""
        user_mgmt_frame = bstrap.Frame(notebook, padding=15)
        notebook.add(user_mgmt_frame, text=' User Management ')

        left_panel = bstrap.Frame(user_mgmt_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        right_panel = bstrap.Frame(user_mgmt_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        reg_frame = bstrap.LabelFrame(left_panel, text="Register New User", padding=15)
        reg_frame.pack(fill=tk.X, pady=(0, 20))
        bstrap.Label(reg_frame, text="User ID:").grid(row=0, column=0, padx=5, pady=10, sticky=tk.W)
        self.user_id_entry = bstrap.Entry(reg_frame, width=40)
        self.user_id_entry.grid(row=0, column=1, padx=5, pady=10)
        bstrap.Label(reg_frame, text="User Name:").grid(row=1, column=0, padx=5, pady=10, sticky=tk.W)
        self.user_name_entry = bstrap.Entry(reg_frame, width=40)
        self.user_name_entry.grid(row=1, column=1, padx=5, pady=10)

        schedule_frame = bstrap.LabelFrame(reg_frame, text="Schedule (Optional)")
        schedule_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")

        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        self.schedule_day_vars = {day: tk.BooleanVar() for day in days}
        day_frame = bstrap.Frame(schedule_frame)
        day_frame.pack(pady=5, anchor='center')
        bstrap.Label(day_frame, text="Days:").pack(side=tk.LEFT, padx=(0, 10))
        for day in days:
            bstrap.Checkbutton(day_frame, text=day, variable=self.schedule_day_vars[day]).pack(side=tk.LEFT, padx=3)

        time_frame = bstrap.Frame(schedule_frame)
        time_frame.pack(pady=5, anchor='center')
        bstrap.Label(time_frame, text="Time In (HH:MM AM/PM):").pack(side=tk.LEFT, padx=5)
        self.schedule_time_in_entry = bstrap.Entry(time_frame, width=12)
        self.schedule_time_in_entry.pack(side=tk.LEFT)
        bstrap.Label(time_frame, text="Time Out (HH:MM AM/PM):").pack(side=tk.LEFT, padx=(15,5))
        self.schedule_time_out_entry = bstrap.Entry(time_frame, width=12)
        self.schedule_time_out_entry.pack(side=tk.LEFT)

        self.register_button = bstrap.Button(reg_frame, text=" Start Interactive Registration", image=self.add_user_icon, compound=tk.LEFT, command=self.register_user)
        self.register_button.grid(row=3, column=0, columnspan=2, pady=20)
        
        self.save_edit_button = bstrap.Button(reg_frame, text=" Save Changes", image=self.save_icon, compound=tk.LEFT, command=self.save_user_edits)
        self.cancel_edit_button = bstrap.Button(reg_frame, text=" Cancel Edit", image=self.cancel_icon, compound=tk.LEFT, command=self.cancel_edit_mode, bootstyle="danger")

        user_list_frame = bstrap.LabelFrame(left_panel, text="Registered Users", padding=15)
        user_list_frame.pack(fill=tk.BOTH, expand=True)
        self.user_tree = bstrap.Treeview(user_list_frame, columns=('ID', 'Name'), show='headings')
        self.user_tree.heading('ID', text='ID')
        self.user_tree.heading('Name', text='Name')
        self.user_tree.column('ID', anchor='center', width=100)
        self.user_tree.column('Name', anchor='w')
        self.user_tree.pack(fill=tk.BOTH, expand=True)
        self.user_tree.bind('<<TreeviewSelect>>', self.update_user_details_view)
        
        details_frame = bstrap.LabelFrame(right_panel, text="User Details", padding=15)
        details_frame.pack(fill=tk.Y)
        self.user_photo_label = bstrap.Label(details_frame, background="#E5E7EB", width=28)
        self.user_photo_label.pack(pady=10, padx=10)
        self.details_id_label = bstrap.Label(details_frame, text="ID: -", font=('Inter', 11, 'bold'))
        self.details_id_label.pack(pady=(10,0))
        self.details_name_label = bstrap.Label(details_frame, text="Name: -", font=('Inter', 10))
        self.details_name_label.pack(pady=(0,5))
        self.details_schedule_label = bstrap.Label(details_frame, text="Schedule: -", font=('Inter', 9), justify=tk.LEFT)
        self.details_schedule_label.pack(pady=(0,15))
        
        bstrap.Button(details_frame, text=" Edit User", image=self.edit_user_icon, compound=tk.LEFT, command=self.edit_user).pack(pady=5, fill=tk.X)
        bstrap.Button(details_frame, text=" Delete User", image=self.delete_user_icon, compound=tk.LEFT, command=self.delete_user, bootstyle="danger").pack(pady=5, fill=tk.X)
        bstrap.Button(details_frame, text=" Refresh List", image=self.refresh_icon, compound=tk.LEFT, command=self.load_users, bootstyle="info-outline").pack(pady=(15, 5), fill=tk.X)

    def create_history_tab(self, notebook):
        """Creates the 'Attendance History' tab for viewing and filtering past records."""
        history_frame = bstrap.Frame(notebook, padding=15)
        notebook.add(history_frame, text=' Attendance History ')
        
        # --- NEW: Replaced Year/Month filter with a Date Range filter ---
        filter_frame = bstrap.LabelFrame(history_frame, text="Filter by Date Range", padding=15)
        filter_frame.pack(fill=tk.X, pady=(0, 10))

        bstrap.Label(filter_frame, text="Start Date:").pack(side=tk.LEFT, padx=(0, 5))
        self.history_start_date_entry = bstrap.DateEntry(filter_frame, bootstyle="primary", dateformat="%Y-%m-%d")
        self.history_start_date_entry.entry.config(font=('Inter', 10))
        # Set default start date to 30 days ago
        self.history_start_date_entry.set_date(datetime.now() - timedelta(days=30))
        self.history_start_date_entry.pack(side=tk.LEFT, padx=5)

        bstrap.Label(filter_frame, text="End Date:").pack(side=tk.LEFT, padx=(15, 5))
        self.history_end_date_entry = bstrap.DateEntry(filter_frame, bootstyle="primary", dateformat="%Y-%m-%d")
        self.history_end_date_entry.entry.config(font=('Inter', 10))
        self.history_end_date_entry.pack(side=tk.LEFT, padx=5)

        bstrap.Button(filter_frame, text="Load Data", command=self.load_historical_data).pack(side=tk.LEFT, padx=20)
        bstrap.Button(filter_frame, text="Export this View", image=self.export_icon, compound=tk.LEFT, command=self.export_history_view, bootstyle="info-outline").pack(side=tk.RIGHT, padx=5)

        history_data_frame = bstrap.Frame(history_frame)
        history_data_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.history_tree = bstrap.Treeview(history_data_frame, columns=('ID', 'Name', 'Date', 'TimeIn', 'TimeOut'), show='headings')
        
        for col in self.history_tree['columns']:
             self.history_tree.heading(col, text=col.replace('Time', 'Time '))
             self.history_tree.column(col, anchor='center', width=120)
        self.history_tree.column('ID', width=50)
        self.history_tree.column('Name', anchor='w', width=150)

        self.history_tree.pack(fill=tk.BOTH, expand=True)

    def create_export_tab(self, notebook):
        """Creates the 'Export' tab for downloading attendance logs."""
        export_frame = bstrap.Frame(notebook, padding=20)
        notebook.add(export_frame, text='   Export Logs   ')
        
        # --- Frame for exporting ALL data ---
        full_export_frame = bstrap.LabelFrame(export_frame, text="Full Log Export", padding=15)
        full_export_frame.pack(fill=tk.X, pady=10)
        
        bstrap.Label(full_export_frame, text="Export the complete history for all users.").pack(pady=(0, 10), anchor='w')
        bstrap.Button(full_export_frame, text="Export Full Attendance Summary (CSV)", image=self.export_icon, compound=tk.LEFT, command=self.export_full_attendance).pack(pady=5, anchor='w')
        bstrap.Button(full_export_frame, text="Export Full Detailed Scan Log (CSV)", image=self.export_icon, compound=tk.LEFT, command=self.export_scan_log, bootstyle="info").pack(pady=10, anchor='w')
        
        # --- NEW: Frame for exporting by date range ---
        range_export_frame = bstrap.LabelFrame(export_frame, text="Custom Date Range Export", padding=15)
        range_export_frame.pack(fill=tk.X, pady=10)

        date_controls_frame = bstrap.Frame(range_export_frame)
        date_controls_frame.pack(fill=tk.X, pady=(0, 15))

        bstrap.Label(date_controls_frame, text="Start Date:").pack(side=tk.LEFT, padx=(0, 5))
        self.export_start_date_entry = bstrap.DateEntry(date_controls_frame, bootstyle="primary", dateformat="%Y-%m-%d")
        self.export_start_date_entry.entry.config(font=('Inter', 10))
        self.export_start_date_entry.set_date(datetime.now() - timedelta(days=7)) # Default to one week ago
        self.export_start_date_entry.pack(side=tk.LEFT, padx=5)

        bstrap.Label(date_controls_frame, text="End Date:").pack(side=tk.LEFT, padx=(20, 5))
        self.export_end_date_entry = bstrap.DateEntry(date_controls_frame, bootstyle="primary", dateformat="%Y-%m-%d")
        self.export_end_date_entry.entry.config(font=('Inter', 10))
        self.export_end_date_entry.pack(side=tk.LEFT, padx=5)
        
        bstrap.Button(range_export_frame, text="Export Range Summary (CSV)", image=self.export_icon, compound=tk.LEFT, command=self.export_attendance_range).pack(pady=5, anchor='w')
        bstrap.Button(range_export_frame, text="Export Range Detailed Log (CSV)", image=self.export_icon, compound=tk.LEFT, command=self.export_scan_log_range, bootstyle="info").pack(pady=10, anchor='w')


    def create_settings_tab(self, notebook):
        """Creates the 'Settings' tab for application configuration."""
        settings_frame = bstrap.Frame(notebook, padding=15)
        notebook.add(settings_frame, text='   Settings   ')
        
        settings_panel = bstrap.LabelFrame(settings_frame, text="Application Settings", padding=15)
        settings_panel.pack(fill=tk.X, pady=10)
        
        bstrap.Label(settings_panel, text="Recognition Cooldown (seconds):").grid(row=0, column=0, padx=5, pady=10, sticky=tk.W)
        self.cooldown_var = tk.StringVar(value=str(self.RECOGNITION_COOLDOWN_SECONDS))
        bstrap.Entry(settings_panel, textvariable=self.cooldown_var, width=20).grid(row=0, column=1, padx=5, pady=10)
        
        bstrap.Button(settings_panel, text="Save Settings", command=self.apply_settings).grid(row=2, column=0, columnspan=2, pady=20)
        
        bstrap.Label(settings_panel, text="Note: Changes will be applied after restarting the application.", font=('Inter', 9, 'italic')).grid(row=3, column=0, columnspan=2)

    def apply_settings(self):
        """Applies and saves the new settings."""
        try:
            new_cooldown = int(self.cooldown_var.get())
            if new_cooldown < 0:
                self.show_toast("Invalid Value", "Cooldown must be a positive number.", "warning")
                return
            self.config.set('Settings', 'RecognitionCooldownSeconds', str(new_cooldown))
            self.save_config()
        except ValueError:
            self.show_toast("Invalid Input", "Please enter a valid number for the cooldown.", "warning")
        except Exception as e:
            logging.error(f"Error applying settings: {e}")
            self.show_toast("Error", "Could not apply settings.", "danger")

    def update_scan_status(self, is_scanning):
        """Updates the status label text and color."""
        if is_scanning:
            self.scan_status_label.config(text="Status: Scanning...", style="success.TLabel")
        else:
            self.scan_status_label.config(text="Status: Not Scanning", style="danger.TLabel")

    def update_user_details_view(self, event=None):
        """Displays selected user's photo and details in the side panel."""
        selected_items = self.user_tree.selection()
        if not selected_items:
            self.user_photo_label.config(image='', text="\n\nNo User\nSelected", font=('Inter', 12))
            self.user_photo_label.image = None
            self.details_id_label.config(text="ID: -")
            self.details_name_label.config(text="Name: -")
            self.details_schedule_label.config(text="Schedule: -")
            return

        user_id, user_name = self.user_tree.item(selected_items[0])['values']
        self.details_id_label.config(text=f"ID: {user_id}")
        self.details_name_label.config(text=f"Name: {user_name}")
        
        schedule_text = "Schedule: Not Set"
        try:
            students_df = self.safe_read_csv(self.students_file)
            if students_df is not None:
                # Ensure ID column is integer for correct comparison
                students_df['ID'] = pd.to_numeric(students_df['ID'], errors='coerce').dropna().astype(int)
                user_data_row = students_df[students_df['ID'] == int(user_id)]
                if not user_data_row.empty:
                    user_data = user_data_row.iloc[0]
                    days = user_data.get('ScheduleDays')
                    time_in = user_data.get('ScheduleTimeIn')
                    time_out = user_data.get('ScheduleTimeOut')
                    if pd.notna(days) and days and pd.notna(time_in) and time_in:
                        schedule_text = f"Days: {days}\nTime: {time_in} - {time_out or 'N/A'}"
        except Exception as e:
            logging.warning(f"Could not retrieve schedule for user {user_id}: {e}")
        self.details_schedule_label.config(text=schedule_text)

        photo_path = os.path.join('registered_faces', f"{user_id}.jpg")
        if os.path.exists(photo_path):
            try:
                with Image.open(photo_path) as img:
                    img.thumbnail((200, 200))
                    photo = ImageTk.PhotoImage(img)
                    self.user_photo_label.config(image=photo, text="")
                    self.user_photo_label.image = photo
            except Exception as e:
                logging.error(f"Error loading image {photo_path}: {e}")
                self.user_photo_label.config(image='', text="\n\nImage Not\nFound", font=('Inter', 12))
        else:
            self.user_photo_label.config(image='', text="\n\nNo Image\nAvailable", font=('Inter', 12))
            self.user_photo_label.image = None

    def _calculate_head_tilt_angle(self, landmarks):
        """Calculates the head tilt angle."""
        nose_bridge = landmarks['nose_bridge'][0]
        chin = landmarks['chin'][8]
        dx = chin[0] - nose_bridge[0]
        dy = chin[1] - nose_bridge[1]
        return np.degrees(np.arctan2(dx, dy))

    def _validate_and_format_time(self, time_str):
        """Validates time is in HH:MM AM/PM format and standardizes it."""
        if not time_str:
            return "" 
        
        match = re.match(r'^(0?[1-9]|1[0-2]):([0-5]\d)\s*(AM|PM)$', time_str, re.IGNORECASE)
        if match:
            hour, minute, period = match.groups()
            return f"{int(hour):02d}:{minute} {period.upper()}"
        return None

    def register_user(self):
        """Handles the interactive user registration process with liveness checks."""
        user_id = self.user_id_entry.get().strip()
        user_name = self.user_name_entry.get().strip()

        if not user_id or not user_name:
            self.show_toast("Registration Error", "User ID and Name cannot be empty.", "danger")
            return
        
        schedule_time_in = self._validate_and_format_time(self.schedule_time_in_entry.get().strip())
        schedule_time_out = self._validate_and_format_time(self.schedule_time_out_entry.get().strip())
        
        if self.schedule_time_in_entry.get().strip() and schedule_time_in is None:
            self.show_toast("Invalid Format", "Time In must be HH:MM AM/PM (e.g., 09:00 AM)", "warning")
            return
        if self.schedule_time_out_entry.get().strip() and schedule_time_out is None:
            self.show_toast("Invalid Format", "Time Out must be HH:MM AM/PM (e.g., 05:00 PM)", "warning")
            return
            
        try:
            students_df = self.safe_read_csv(self.students_file)
            if students_df is not None:
                 # Ensure ID column is treated correctly for comparison
                students_df['ID'] = pd.to_numeric(students_df['ID'], errors='coerce')
                if not students_df[students_df['ID'] == int(user_id)].empty:
                    self.show_toast("Registration Error", "This User ID already exists.", "danger")
                    return
        except ValueError:
            self.show_toast("Invalid ID", "User ID must be a number.", "warning")
            return
        except Exception as e:
            logging.error(f"Error checking existing user ID: {e}")

        if not self.camera_options:
            self.show_toast("Camera Error", "No camera available for registration.", "danger")
            return
            
        cap = cv2.VideoCapture(int(self.camera_selection_var.get()) + cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.show_toast("Camera Error", "Could not open webcam.", "danger")
            return
        
        step = "LOOK_STRAIGHT"
        blink_counter = 0
        forward_facing_frame = None
        instruction = "Position face in oval and Press Spacebar"
        accent_color_bgr = (0, 215, 255)

        cv2.namedWindow("Interactive Registration")

        while True:
            ret, frame = cap.read()
            if not ret: break
            h, w, _ = frame.shape
            
            center_x, center_y = w // 2, h // 2
            cv2.ellipse(frame, (center_x, center_y), (w//4, h//3), 0, 0, 360, accent_color_bgr, 2)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            (text_w, text_h), _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(frame, instruction, (center_x - text_w // 2, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, accent_color_bgr, 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if len(face_locations) == 1:
                landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)[0]
                
                if step == "LOOK_STRAIGHT":
                    if cv2.waitKey(1) & 0xFF == 32:
                        forward_facing_frame = frame.copy()
                        step = "BLINK"
                        instruction = "Great! Now blink three times."
                
                elif step == "BLINK":
                    ear = (self.eye_aspect_ratio(landmarks['left_eye']) + self.eye_aspect_ratio(landmarks['right_eye'])) / 2.0
                    if ear < self.EYE_AR_THRESH: blink_counter += 1
                    else:
                        if blink_counter >= self.EYE_AR_CONSEC_FRAMES_REGISTER:
                            step = "TILT_LEFT"
                            instruction = f"Tilt head left > {self.HEAD_TILT_THRESH} degrees"
                        blink_counter = 0
                
                elif "TILT" in step:
                    angle = self._calculate_head_tilt_angle(landmarks)
                    if step == "TILT_LEFT" and angle > self.HEAD_TILT_THRESH:
                        step = "TILT_RIGHT"
                        instruction = f"Perfect! Now tilt head right > {self.HEAD_TILT_THRESH} degrees"
                    elif step == "TILT_RIGHT" and angle < -self.HEAD_TILT_THRESH:
                        step = "DONE"
                        instruction = "All set! Registering..."

            cv2.imshow("Interactive Registration", frame)
            
            if step == "DONE":
                encodings = face_recognition.face_encodings(cv2.cvtColor(forward_facing_frame, cv2.COLOR_BGR2RGB))
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_ids.append(user_id)
                    self.save_known_faces()
                    
                    selected_days = ",".join([day for day, var in self.schedule_day_vars.items() if var.get()])
                    
                    with open(self.students_file, 'a', newline='') as f:
                        csv.writer(f).writerow([user_id, user_name, selected_days, schedule_time_in, schedule_time_out])
                    
                    cv2.imwrite(os.path.join('registered_faces', f"{user_id}.jpg"), forward_facing_frame)
                    
                    self.show_toast("Success", f"User {user_name} registered successfully.", "success")
                    self.user_id_entry.delete(0, tk.END)
                    self.user_name_entry.delete(0, tk.END)
                    for var in self.schedule_day_vars.values(): var.set(False)
                    self.schedule_time_in_entry.delete(0, tk.END)
                    self.schedule_time_out_entry.delete(0, tk.END)
                    self.load_users()
                    break
            
            if cv2.waitKey(1) & 0xFF == 27: break
        
        cap.release()
        cv2.destroyAllWindows()

    def start_scanning(self):
        """Starts the camera and the GUI display loop from the main thread."""
        if self.scanning:
            return
        
        try:
            camera_index = int(self.camera_selection_var.get())
            self.scanning = True
            
            self.start_scan_button.config(state=tk.DISABLED)
            self.stop_scan_button.config(state=tk.NORMAL)
            self.update_scan_status(True)
            self.camera_label.config(text="\n\nInitializing Camera...")

            self.camera_thread = threading.Thread(target=self._camera_thread_loop, args=(camera_index,), daemon=True)
            self.camera_thread.start()

            self.root.after(20, self.scan_loop)

        except (ValueError, IndexError):
            self.show_toast("Camera Error", "Invalid or no camera selected.", "danger")

    def stop_scanning(self):
        """Stops the camera feed and signals all threads to stop."""
        self.scanning = False
        
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

        self.start_scan_button.config(state=tk.NORMAL)
        self.stop_scan_button.config(state=tk.DISABLED)
        self.camera_label.config(image='')
        if not self.camera_options:
             self.camera_label.config(text="\n\nNo Camera Found")
        self.update_scan_status(False)

    def _camera_thread_loop(self, camera_index):
        """Handles camera initialization and frame grabbing in a separate thread."""
        self.cap = cv2.VideoCapture(camera_index + cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.root.after(0, self.show_toast, "Camera Error", f"Could not open camera {camera_index}.", "danger")
            self.root.after(0, self.stop_scanning)
            return

        self.processing_thread = threading.Thread(target=self._processing_thread_loop, daemon=True)
        self.processing_thread.start()
        
        while self.scanning:
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(0, self.show_toast, "Camera Error", "Failed to capture frame.", "danger")
                self.scanning = False
                break
            
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            time.sleep(1/60)

        if self.cap:
            self.cap.release()
        self.cap = None

    def scan_loop(self):
        """The main GUI thread loop for displaying the camera feed."""
        if not self.scanning:
            return

        with self.frame_lock:
            frame_to_display = self.current_frame.copy() if self.current_frame is not None else None
        
        if frame_to_display is not None:
            for i, (top, right, bottom, left) in enumerate(self.last_known_locations):
                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                if i < len(self.last_known_names):
                    name_info = self.last_known_names[i]
                    name, face_id = name_info["name"], name_info["id"]
                    
                    color = (0, 0, 255) # Red for unknown
                    if name != "Unknown":
                        color = (0, 255, 0) if self.live_blink_counters.get(face_id, 0) > 0 else (0, 215, 255)

                    cv2.rectangle(frame_to_display, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(frame_to_display, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    cv2.putText(frame_to_display, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (27, 27, 27), 1)
            
            img = Image.fromarray(cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.config(image=imgtk)
        
        self.root.after(30, self.scan_loop)

    def _processing_thread_loop(self):
        """The background thread for heavy face recognition processing with frame skipping."""
        students_df = self.safe_read_csv(self.students_file)
        if students_df is None:
            students_df = pd.DataFrame(columns=['ID', 'Name'])
        else:
            # Ensure ID column is numeric for later comparisons
            students_df['ID'] = pd.to_numeric(students_df['ID'], errors='coerce')

        while self.scanning:
            with self.frame_lock:
                frame_to_process = self.current_frame.copy() if self.current_frame is not None else None

            if frame_to_process is None:
                time.sleep(0.1)
                continue
            
            self.frame_counter += 1
            if self.frame_counter % self.PROCESS_EVERY_N_FRAMES == 0:
                small_frame = cv2.resize(frame_to_process, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                landmarks = face_recognition.face_landmarks(rgb_small_frame, face_locations)
                
                current_names = []
                for i, enc in enumerate(face_encodings):
                    matches = face_recognition.compare_faces(self.known_face_encodings, enc, tolerance=0.5)
                    name, face_id = "Unknown", None
                    
                    if any(matches):
                        face_distances = face_recognition.face_distance(self.known_face_encodings, enc)
                        best_match_idx = np.argmin(face_distances)
                        if matches[best_match_idx]:
                            face_id = self.known_face_ids[best_match_idx]
                            
                            ear = (self.eye_aspect_ratio(landmarks[i]['left_eye']) + self.eye_aspect_ratio(landmarks[i]['right_eye'])) / 2.0
                            if ear < self.EYE_AR_THRESH:
                                self.live_blink_counters[face_id] = self.live_blink_counters.get(face_id, 0) + 1
                            else:
                                if self.live_blink_counters.get(face_id, 0) >= self.EYE_AR_CONSEC_FRAMES_ATTENDANCE:
                                    self.root.after(0, self.log_attendance, face_id)
                                self.live_blink_counters[face_id] = 0
                            
                            try:
                                # face_id is a string, students_df['ID'] is numeric
                                user_info = students_df[students_df['ID'] == int(face_id)]
                                if not user_info.empty: name = user_info.iloc[0]['Name']
                            except (ValueError, KeyError): pass
                                
                    current_names.append({"name": name, "id": face_id})
                
                self.last_known_locations = face_locations
                self.last_known_names = current_names
            
            time.sleep(0.01)

    def _get_and_validate_dates(self, start_date_entry, end_date_entry):
        """Helper function to get and validate date range from DateEntry widgets."""
        try:
            start_date_str = start_date_entry.entry.get()
            end_date_str = end_date_entry.entry.get()
            
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

            if start_date > end_date:
                self.show_toast("Date Error", "Start date cannot be after the end date.", "warning")
                return None, None
            
            return start_date, end_date
        except (ValueError, TypeError):
            self.show_toast("Date Error", "Please enter valid start and end dates.", "warning")
            return None, None

    def load_historical_data(self):
        """Loads and displays attendance data based on the selected date range."""
        for i in self.history_tree.get_children(): self.history_tree.delete(i)
            
        start_date, end_date = self._get_and_validate_dates(self.history_start_date_entry, self.history_end_date_entry)
        if not start_date:
            return
        
        try:
            df = self.safe_read_csv(self.attendance_file)
            if df is None:
                self.show_toast("No Data", "Attendance file is empty or missing.", "info")
                return

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
            
            # Filter based on the selected date range (inclusive)
            mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
            filtered_df = df.loc[mask]
            
            if filtered_df.empty:
                self.show_toast("No Data", f"No records found for the selected date range.", "info")
                return
            
            filtered_df['Date'] = filtered_df['Date'].dt.strftime('%Y-%m-%d')
            filtered_df = filtered_df.fillna('---')
            for _, row in filtered_df.iloc[::-1].iterrows():
                self.history_tree.insert("", tk.END, values=list(row))
        except Exception as e:
            self.show_toast("Error", f"Could not load data: {e}", "danger")
            logging.error(f"Failed to load historical data: {e}")

    def export_history_view(self):
        """Exports the currently filtered view of attendance history to a CSV file."""
        start_date, end_date = self._get_and_validate_dates(self.history_start_date_entry, self.history_end_date_entry)
        if not start_date:
            return
            
        try:
            df = self.safe_read_csv(self.attendance_file)
            if df is None:
                self.show_toast("Export Error", "Attendance file is empty or missing.", "warning")
                return

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)

            mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
            filtered_df = df.loc[mask]
            
            if filtered_df.empty:
                self.show_toast("Export Error", "No data in the current view to export.", "warning")
                return
            
            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv", filetypes=[("CSV files", "*.csv")],
                title="Save Filtered Log",
                initialfile=f"attendance_log_{start_str}_to_{end_str}.csv"
            )
            
            if save_path:
                self.safe_save_csv(filtered_df, save_path)
                self.show_toast("Export Successful", f"Data saved to {os.path.basename(save_path)}", "success")
        except Exception as e:
            self.show_toast("Export Error", f"Failed to export data: {e}", "danger")
            logging.error(f"Failed to export history view: {e}")

    def show_toast(self, title, message, bootstyle="success", duration=4500):
        """Displays a temporary toast notification."""
        icon = self.toast_icons.get(bootstyle)
        ToastNotification(title=title, message=message, duration=duration, bootstyle=bootstyle,
                          position=(20, 20, 'ne'), icon=icon, alert=True).show_toast()

    def eye_aspect_ratio(self, eye):
        """Calculates the Eye Aspect Ratio (EAR) for blink detection."""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def edit_user(self):
        """Enables 'edit mode' by populating the registration form with the selected user's data."""
        selected_item = self.user_tree.selection()
        if not selected_item:
            self.show_toast("Edit Error", "Please select a user to edit.", "warning")
            return
        
        self.editing_user_id, old_name = self.user_tree.item(selected_item, 'values')
        
        try:
            students_df = self.safe_read_csv(self.students_file)
            if students_df is None: return
            
            students_df['ID'] = pd.to_numeric(students_df['ID'], errors='coerce').dropna().astype(int)
            user_data = students_df[students_df['ID'] == int(self.editing_user_id)].iloc[0]

            self.user_id_entry.delete(0, tk.END)
            self.user_id_entry.insert(0, self.editing_user_id)
            self.user_id_entry.config(state='readonly')

            self.user_name_entry.delete(0, tk.END)
            self.user_name_entry.insert(0, old_name)

            days_str = user_data.get('ScheduleDays', '')
            days = days_str.split(',') if pd.notna(days_str) and days_str else []
            for day, var in self.schedule_day_vars.items():
                var.set(day in days)
            
            self.schedule_time_in_entry.delete(0, tk.END)
            self.schedule_time_in_entry.insert(0, user_data.get('ScheduleTimeIn', '') or '')

            self.schedule_time_out_entry.delete(0, tk.END)
            self.schedule_time_out_entry.insert(0, user_data.get('ScheduleTimeOut', '') or '')

            self.register_button.grid_remove()
            self.save_edit_button.grid(row=3, column=0, columnspan=2, pady=20, sticky='ew')
            self.cancel_edit_button.grid(row=4, column=0, columnspan=2, pady=(0,10), sticky='ew')
            
            self.notebook.select(1) # Switch to User Management tab
            self.user_name_entry.focus()
        except Exception as e:
            self.show_toast("Error", f"Could not load user data for editing: {e}", "danger")
            logging.error(f"Failed to enter edit mode for user {self.editing_user_id}: {e}")
            self.cancel_edit_mode()

    def save_user_edits(self):
        """Saves the modified user data from the form."""
        if not self.editing_user_id:
            return

        new_name = self.user_name_entry.get().strip()
        if not new_name:
            self.show_toast("Save Error", "User Name cannot be empty.", "danger")
            return

        schedule_time_in = self._validate_and_format_time(self.schedule_time_in_entry.get().strip())
        schedule_time_out = self._validate_and_format_time(self.schedule_time_out_entry.get().strip())

        if self.schedule_time_in_entry.get().strip() and schedule_time_in is None:
            self.show_toast("Invalid Format", "Time In must be HH:MM AM/PM", "warning")
            return
        if self.schedule_time_out_entry.get().strip() and schedule_time_out is None:
            self.show_toast("Invalid Format", "Time Out must be HH:MM AM/PM", "warning")
            return

        try:
            user_id_int = int(self.editing_user_id)
            selected_days = ",".join([day for day, var in self.schedule_day_vars.items() if var.get()])
            
            s_df = self.safe_read_csv(self.students_file)
            if s_df is not None:
                s_df['ID'] = pd.to_numeric(s_df['ID'], errors='coerce')
                idx = s_df.index[s_df['ID'] == user_id_int].tolist()
                if idx:
                    s_df.loc[idx[0], ['Name', 'ScheduleDays', 'ScheduleTimeIn', 'ScheduleTimeOut']] = [new_name, selected_days, schedule_time_in, schedule_time_out]
                    self.safe_save_csv(s_df, self.students_file)
            
            a_df = self.safe_read_csv(self.attendance_file)
            if a_df is not None and not a_df.empty:
                a_df['ID'] = pd.to_numeric(a_df['ID'], errors='coerce')
                a_df.loc[a_df['ID'] == user_id_int, 'Name'] = new_name
                self.safe_save_csv(a_df, self.attendance_file)
                    
            self.show_toast("Success", f"User {new_name}'s details updated.", "success")
            self.load_users()
            self.load_attendance()
            self.cancel_edit_mode()
        except Exception as e:
            self.show_toast("Save Error", f"An error occurred: {e}", "danger")
            logging.error(f"Error saving user edits: {e}")

    def cancel_edit_mode(self):
        """Exits 'edit mode' and clears the registration form."""
        self.editing_user_id = None
        
        self.user_id_entry.config(state='normal')
        self.user_id_entry.delete(0, tk.END)
        self.user_name_entry.delete(0, tk.END)
        for var in self.schedule_day_vars.values():
            var.set(False)
        self.schedule_time_in_entry.delete(0, tk.END)
        self.schedule_time_out_entry.delete(0, tk.END)
        
        self.save_edit_button.grid_remove()
        self.cancel_edit_button.grid_remove()
        self.register_button.grid(row=3, column=0, columnspan=2, pady=20)

    def delete_user(self):
        """Deletes a selected user from the system."""
        selected_item = self.user_tree.selection()
        if not selected_item:
            self.show_toast("Deletion Error", "Please select a user.", "warning")
            return
            
        user_id_str, user_name = self.user_tree.item(selected_item, 'values')
        
        if not messagebox.askyesno("Confirm Deletion", f"Delete {user_name} (ID: {user_id_str})? This removes the user and photo permanently. Attendance history will remain."):
            return
            
        try:
            user_id_int = int(user_id_str)
            df = self.safe_read_csv(self.students_file)
            if df is not None:
                df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
                df = df[df['ID'] != user_id_int]
                self.safe_save_csv(df, self.students_file)
            
            if self.known_face_ids:
                indices_to_keep = [i for i, fid in enumerate(self.known_face_ids) if fid != user_id_str]
                self.known_face_encodings = [self.known_face_encodings[i] for i in indices_to_keep]
                self.known_face_ids = [self.known_face_ids[i] for i in indices_to_keep]
                self.save_known_faces()
            
            img_path = os.path.join('registered_faces', f"{user_id_str}.jpg")
            if os.path.exists(img_path): os.remove(img_path)
                
            self.show_toast("Success", f"User {user_name} has been removed.", "success")
            self.load_users()
        except Exception as e:
            self.show_toast("Deletion Error", f"An error occurred: {e}", "danger")
            logging.error(f"Error deleting user: {e}")

    def log_attendance(self, face_id):
        """Logs scans and plays sounds for Time In, Time Out, and Late status."""
        now = datetime.now()
        if (now - self.last_recognition_times.get(face_id, datetime.min)).total_seconds() < self.RECOGNITION_COOLDOWN_SECONDS:
            return

        try:
            s_df = self.safe_read_csv(self.students_file)
            if s_df is None: return
            
            s_df['ID'] = pd.to_numeric(s_df['ID'], errors='coerce')
            user_info_row = s_df[s_df['ID'] == int(face_id)]
            if user_info_row.empty: return
            
            user_info = user_info_row.iloc[0]
            user_name = user_info['Name']
            date = now.strftime('%Y-%m-%d')
            time_str = now.strftime('%I:%M:%S %p')
            
            with open(self.scan_log_file, 'a', newline='') as f:
                csv.writer(f).writerow([face_id, user_name, date, time_str])

            a_df = self.safe_read_csv(self.attendance_file)
            if a_df is None:
                a_df = pd.DataFrame(columns=['ID', 'Name', 'Date', 'TimeIn', 'TimeOut'])

            a_df['ID'] = pd.to_numeric(a_df['ID'], errors='coerce')
            today_record = a_df[(a_df['ID'] == int(face_id)) & (a_df['Date'] == date)]
            
            if today_record.empty:
                is_late = False
                schedule_days = user_info.get('ScheduleDays')
                schedule_time_in_str = user_info.get('ScheduleTimeIn')

                if pd.notna(schedule_days) and now.strftime('%a') in schedule_days and pd.notna(schedule_time_in_str) and schedule_time_in_str:
                    try:
                        schedule_time = datetime.strptime(schedule_time_in_str, '%I:%M %p').time()
                        if now.time() > schedule_time:
                            is_late = True
                    except ValueError:
                        logging.error(f"Could not parse schedule time '{schedule_time_in_str}' for user {face_id}")

                new_entry = pd.DataFrame([{'ID': int(face_id), 'Name': user_name, 'Date': date, 'TimeIn': time_str, 'TimeOut': ''}])
                a_df = pd.concat([a_df, new_entry], ignore_index=True)
                
                toast_msg = f"{user_name} clocked in at {time_str}."
                sound_type = 'in'
                if is_late:
                    toast_msg = f"{user_name} clocked in LATE at {time_str}."
                    sound_type = 'late'
                
                self.show_toast("Time In" if not is_late else "Late", toast_msg, "success" if not is_late else "warning")
                self.play_sound(sound_type)
            else:
                a_df.loc[today_record.index[0], 'TimeOut'] = time_str
                self.show_toast("Scan Recorded", f"{user_name}'s new scan time is {time_str}.", "info")
                self.play_sound('out')
                
            self.safe_save_csv(a_df, self.attendance_file)
            self.load_attendance()
            self.last_recognition_times[face_id] = now
        
        except Exception as e:
            self.show_toast("Logging Error", f"An error occurred: {e}", "danger")
            logging.error(f"Error logging attendance: {e}")

    def load_attendance(self):
        """Loads and displays the attendance summary in the live log."""
        for i in self.tree.get_children(): self.tree.delete(i)
        df = self.safe_read_csv(self.attendance_file)
        if df is not None and not df.empty:
            df = df.fillna('---')
            for _, row in df.iloc[::-1].iterrows():
                self.tree.insert("", tk.END, values=list(row))

    def load_users(self):
        """Loads and displays all registered users."""
        for i in self.user_tree.get_children(): self.user_tree.delete(i)
        df = self.safe_read_csv(self.students_file)
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                self.user_tree.insert("", tk.END, values=(row['ID'], row['Name']))
        self.update_user_details_view()

    def export_full_attendance(self):
        """Exports the entire attendance summary log to a user-specified CSV file."""
        self.export_attendance_range(is_full_export=True)

    def export_scan_log(self):
        """Exports the entire detailed scan log, formatted for readability."""
        self.export_scan_log_range(is_full_export=True)

    def export_attendance_range(self, is_full_export=False):
        """Exports attendance summary for a specific date range, or all data if is_full_export is True."""
        start_date, end_date = None, None
        if not is_full_export:
            start_date, end_date = self._get_and_validate_dates(self.export_start_date_entry, self.export_end_date_entry)
            if not start_date:
                return
        
        try:
            df = self.safe_read_csv(self.attendance_file)
            if df is None or df.empty:
                self.show_toast("Export Error", "No attendance summary data to export.", "warning")
                return
            
            if is_full_export:
                filtered_df = df
                initial_file = f"full_attendance_summary_{datetime.now().strftime('%Y-%m-%d')}.csv"
            else:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df.dropna(subset=['Date'], inplace=True)
                mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
                filtered_df = df.loc[mask]
                start_str, end_str = start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')
                initial_file = f"attendance_summary_{start_str}_to_{end_str}.csv"

            if filtered_df.empty:
                self.show_toast("Export Error", "No data found for the selected date range.", "warning")
                return

            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv", filetypes=[("CSV files", "*.csv")],
                title="Save Attendance Summary As", initialfile=initial_file
            )
            
            if save_path:
                self.safe_save_csv(filtered_df, save_path)
                self.show_toast("Export Successful", f"Summary saved to {os.path.basename(save_path)}", "success")
        except Exception as e:
            self.show_toast("Export Error", f"Failed to export data: {e}", "danger")
            logging.error(f"Error exporting attendance summary: {e}")

    def export_scan_log_range(self, is_full_export=False):
        """Exports the detailed scan log for a date range, or all data if is_full_export is True."""
        start_date, end_date = None, None
        if not is_full_export:
            start_date, end_date = self._get_and_validate_dates(self.export_start_date_entry, self.export_end_date_entry)
            if not start_date:
                return

        try:
            df = self.safe_read_csv(self.scan_log_file)
            if df is None or df.empty:
                self.show_toast("Export Error", "No detailed scan data to export.", "warning")
                return
            
            # --- Data Cleaning and Preparation ---
            df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['ID', 'Date'], inplace=True)
            df['ID'] = df['ID'].astype(int)
            
            if is_full_export:
                filtered_df = df
                start_str = "full_log"
                end_str = datetime.now().strftime('%Y-%m-%d')
            else:
                mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
                filtered_df = df.loc[mask]
                start_str = start_date.strftime('%Y%m%d')
                end_str = end_date.strftime('%Y%m%d')

            if filtered_df.empty:
                self.show_toast("Export Error", "No data found for the selected date range.", "warning")
                return
            
            # Continue with processing on the 'filtered_df'
            time_str = filtered_df['Time'].astype(str)
            date_str = filtered_df['Date'].dt.strftime('%Y-%m-%d')
            filtered_df['timestamp'] = pd.to_datetime(date_str + ' ' + time_str, errors='coerce')
            filtered_df.dropna(subset=['timestamp'], inplace=True)
            
            if filtered_df.empty:
                self.show_toast("No Data", "No valid time entries found in the selected range.", "info")
                return

            filtered_df.sort_values(by=['ID', 'timestamp'], inplace=True)
            
            # --- Data Reshaping ---
            filtered_df['FormattedDate'] = filtered_df['timestamp'].dt.strftime('%m/%d/%Y')
            filtered_df['FormattedTime'] = filtered_df['timestamp'].dt.strftime('%I:%M:%S %p')
            filtered_df['scan_num'] = filtered_df.groupby(['ID', 'FormattedDate']).cumcount() + 1
            filtered_df['scan_col_name'] = 'Scan' + filtered_df['scan_num'].astype(str)
            
            reshaped_df = filtered_df.pivot_table(
                index=['ID', 'Name', 'FormattedDate'],
                columns='scan_col_name',
                values='FormattedTime',
                aggfunc='first'
            ).reset_index()
            
            reshaped_df.columns.name = None
            reshaped_df.rename(columns={'FormattedDate': 'Date'}, inplace=True)
            
            base_cols = ['ID', 'Name', 'Date']
            scan_cols = sorted([col for col in reshaped_df.columns if col.startswith('Scan')], 
                               key=lambda c: int(c.replace('Scan', '')))
            final_column_order = base_cols + scan_cols
            for col in final_column_order:
                if col not in reshaped_df.columns:
                    reshaped_df[col] = None
            reshaped_df = reshaped_df[final_column_order]

            if 'Scan1' in reshaped_df.columns:
                reshaped_df['sort_date'] = pd.to_datetime(reshaped_df['Date'], format='%m/%d/%Y')
                reshaped_df['sort_time'] = pd.to_datetime(reshaped_df['Scan1'], format='%I:%M:%S %p', errors='coerce').dt.time
                reshaped_df.sort_values(by=['sort_date', 'sort_time'], inplace=True, na_position='first')
                reshaped_df.drop(columns=['sort_date', 'sort_time'], inplace=True)
            else:
                reshaped_df.sort_values(by=['Date', 'Name'], inplace=True)

            reshaped_df.insert(0, '#', range(1, 1 + len(reshaped_df)))
            
            initial_filename = f"detailed_scan_log_{start_str}_to_{end_str}.csv"
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv", filetypes=[("CSV files", "*.csv")],
                title="Save Detailed Scan Log As", initialfile=initial_filename
            )
            
            if save_path:
                reshaped_df.fillna('---', inplace=True)
                self.safe_save_csv(reshaped_df, save_path)
                self.show_toast("Export Successful", f"Formatted log saved to {os.path.basename(save_path)}", "success")
                
        except Exception as e:
            self.show_toast("Export Error", f"An unexpected error occurred: {e}", "danger")
            logging.error(f"Error exporting detailed scan log: {e}")
            
    def safe_read_csv(self, file_path):
        """Reads a CSV file, falling back to the most recent backup on failure."""
        try:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                return pd.read_csv(file_path, dtype={'ID': str})
            else:
                return None
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logging.error(f"Corruption detected in {file_path}: {e}. Attempting to restore from backup.")
            if self.restore_from_backup(file_path):
                self.show_toast("Data Restored", f"{os.path.basename(file_path)} was restored from backup.", "warning")
                return pd.read_csv(file_path, dtype={'ID': str})
            else:
                self.show_toast("Data Corruption", f"{os.path.basename(file_path)} is corrupted. No valid backup found.", "danger")
                return None

    def safe_save_csv(self, dataframe, file_path):
        """Saves a DataFrame to a CSV file safely."""
        try:
            dataframe.to_csv(file_path, index=False)
        except Exception as e:
            logging.error(f"Failed to save to {file_path}: {e}")
            self.show_toast("Save Error", f"Could not save data to {os.path.basename(file_path)}.", "danger")
            
    def restore_from_backup(self, file_to_restore):
        """Finds the latest backup and restores the specified file."""
        backup_dirs = sorted([d for d in os.listdir('data_backups') if os.path.isdir(os.path.join('data_backups', d))], reverse=True)
        filename = os.path.basename(file_to_restore)
        
        for backup in backup_dirs:
            backup_file_path = os.path.join('data_backups', backup, filename)
            if os.path.exists(backup_file_path):
                try:
                    shutil.copy(backup_file_path, file_to_restore)
                    logging.warning(f"Restored {filename} from backup {backup}.")
                    return True
                except Exception as e:
                    logging.error(f"Failed to restore {filename} from {backup}: {e}")
        return False


if __name__ == "__main__":
    root = bstrap.Window()
    app = FacialRecognitionAttendanceSystem(root)

    def on_closing():
        """Gracefully handle window closing by stopping the camera scan."""
        if app.scanning:
            app.stop_scanning()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
