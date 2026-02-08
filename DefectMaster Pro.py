import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys
import random
import json
import numpy as np
from PIL import Image, ImageTk
import importlib.util
from datetime import datetime
import traceback
import threading
os.environ['PYTHONUNBUFFERED'] = '1'

# Dynamic module import
def import_module_from_path(module_name, file_path):
    if not os.path.exists(file_path):
        messagebox.showerror("Error", f"Module not found: {file_path}\n"
                                      "put following files under this directory:\n"
                                      "- BBox Annotation Tool.py\n- Few-shot Ensemble Learning 090404.py\n- Multi-class Defect Statistics Excel to NPY Converter.py")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Import modules
excel_to_npy_module = import_module_from_path("excel_to_npy", "Multi-class Defect Statistics Excel to NPY Converter.py")
bbox_tool_module = import_module_from_path("bbox_tool", "BBox Annotation Tool.py")
train_module = import_module_from_path("train_module", "Few-shot Ensemble Learning TPED_FF.py")


# stderr redirect
class StderrRedirector:
    def __init__(self, log_text, root):
        self.log_text = log_text
        self.root = root

    def write(self, s):
        if s.strip():
            self.log_text.insert(tk.END, s)
            self.log_text.see(tk.END)
            self.root.update_idletasks()

    def flush(self):
        pass


class DefectTrainingPro:
    def __init__(self, root):
        self.root = root
        self.root.title("DefectMaster Pro - One-Click Multi-Label Training")
        self.root.geometry("900x960")
        self.root.minsize(850, 600)
        self.root.configure(bg='#f0f2f5')

        # Modern style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 10), background='#f0f2f5')
        style.configure('Title.TLabel', font=('Segoe UI', 14, 'bold'), foreground='#2c3e50')
        style.configure('Header.TLabel', font=('Segoe UI', 11, 'bold'), foreground='#34495e')
        style.configure('TButton', padding=8, font=('Segoe UI', 10, 'bold'))
        style.map('TButton', background=[('active', '#3498db')])

        # Variables
        self.image_dir = tk.StringVar()
        self.excel_path = tk.StringVar()
        self.npy_path = tk.StringVar()
        self.json_path = tk.StringVar()
        self.save_dir = tk.StringVar()
        self.class_names = tk.StringVar(value="No Defect,Dislocaton,Bridge,Junction")
        self.num_images = tk.IntVar(value=10)
        self.num_epochs = tk.IntVar(value=150)
        self.batch_size = tk.IntVar(value=32)
        self.learning_rate = tk.DoubleVar(value=0.0005)
        # self.mode = tk.StringVar(value="train")

        # Hyperparameters
        self.augment_multiplier = tk.IntVar(value=3)
        self.intensity_modifier = tk.DoubleVar(value=0.3)
        self.augment_per_class = {}
        self.num_seeds = tk.IntVar(value=5)
        self.test_size = tk.DoubleVar(value=0.20)
        self.val_size = tk.DoubleVar(value=0.20)
        self.early_stop = tk.IntVar(value=15)
        self.dropout_rate = tk.DoubleVar(value=0.4)

        self.create_widgets()
        self.class_names.trace("w", lambda *_: self.update_per_class_augment())
        self.update_per_class_augment()

        # redirect outputs to GUI log
        def gui_print(*args, sep=" ", end="\n"):
            text = sep.join(map(str, args)) + end
            if text.strip():
                self.log_text.insert(tk.END, text)
                self.log_text.see(tk.END)
                self.root.update_idletasks()
                self.root.update()

                save_dir = self.save_dir.get()
                if save_dir and save_dir.strip():
                    log_path = os.path.join(save_dir, "gui_log.txt")
                    try:
                        with open(log_path, 'a', encoding='utf-8') as f:
                            f.write(text)
                    except:
                        pass

        import builtins
        builtins.print = gui_print

        # stderr
        sys.stderr = StderrRedirector(self.log_text, self.root)  # 传入 self.log_text 和 self.root

        print("=" * 68)
        print("DefectMaster Pro Ready!")
        print("All training logs, warnings, and errors will be displayed here in real time.")
        print("=" * 68)

    def create_widgets(self):
        # ================== main rolling area ==================
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True)

        canvas = tk.Canvas(main_container, highlightthickness=0, bg='#f0f2f5')
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # Scrollable content area
        content_frame = ttk.Frame(scrollable_frame)
        content_frame.pack(padx=30, pady=25, fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(content_frame, text="DefectMaster Pro", style='Title.TLabel')
        title.grid(row=0, column=0, columnspan=4, pady=(0, 20))
        subtitle = ttk.Label(content_frame, text="One-Click Multi-Label Defect Classification Training",
                             font=('Segoe UI', 10), foreground='#7f8c8d')
        subtitle.grid(row=1, column=0, columnspan=4, pady=(0, 30))
        row = 2

        # 1. Image Folder
        ttk.Label(content_frame, text="Image Folder:", style='Header.TLabel').grid(row=row, column=0, sticky=tk.W, pady=(15, 5))
        ttk.Entry(content_frame, textvariable=self.image_dir, width=50).grid(row=row, column=1, columnspan=2, padx=(10, 5), sticky=tk.W+tk.E)
        ttk.Button(content_frame, text="Browse", command=self.browse_image_dir).grid(row=row, column=3, padx=5)
        row += 1

        # 2. Class Names
        ttk.Label(content_frame, text="Class Names (comma separated):", style='Header.TLabel').grid(row=row, column=0, sticky=tk.W, pady=(15, 5))
        ttk.Entry(content_frame, textvariable=self.class_names, width=50).grid(row=row, column=1, columnspan=2, padx=(10, 5), sticky=tk.W+tk.E)
        ttk.Label(content_frame, text="e.g. No Defect,Dislocation,Bridge,Junction", foreground="gray", font=('Segoe UI', 9)).grid(row=row, column=3, sticky=tk.W)
        row += 1

        # 3. Excel Label File
        ttk.Label(content_frame, text="Excel Label File:", style='Header.TLabel').grid(row=row, column=0, sticky=tk.W, pady=(15, 5))
        ttk.Entry(content_frame, textvariable=self.excel_path, width=50).grid(row=row, column=1, columnspan=2, padx=(10, 5), sticky=tk.W+tk.E)
        ttk.Button(content_frame, text="Browse", command=self.browse_excel).grid(row=row, column=3, padx=5)
        row += 1

        # 4. Annotation File (optional)
        ttk.Label(content_frame, text="Annotation File (JSON, optional):", style='Header.TLabel').grid(row=row, column=0, sticky=tk.W, pady=(15, 5))
        ttk.Entry(content_frame, textvariable=self.json_path, width=50).grid(row=row, column=1, columnspan=2, padx=(10, 5), sticky=tk.W+tk.E)
        ttk.Button(content_frame, text="Browse", command=self.browse_json).grid(row=row, column=3, padx=5)
        row += 1

        # 5. Save Directory
        ttk.Label(content_frame, text="Experiment Save Path:", style='Header.TLabel').grid(row=row, column=0, sticky=tk.W, pady=(15, 5))
        ttk.Entry(content_frame, textvariable=self.save_dir, width=50).grid(row=row, column=1, columnspan=2, padx=(10, 5), sticky=tk.W+tk.E)
        ttk.Button(content_frame, text="Browse", command=self.browse_save_dir).grid(row=row, column=3, padx=5)
        row += 1

        # 6. Basic Training Settings
        basic_frame = ttk.LabelFrame(content_frame, text=" Basic Training Settings ", padding=15)
        basic_frame.grid(row=row, column=0, columnspan=4, sticky=tk.W+tk.E, pady=20)
        for i in range(6): basic_frame.grid_columnconfigure(i, weight=1)
        ttk.Label(basic_frame, text="Maximum Epochs:").grid(row=0, column=0, padx=10, sticky=tk.E)
        ttk.Entry(basic_frame, textvariable=self.num_epochs, width=10).grid(row=0, column=1, sticky=tk.W, padx=(0,20))
        ttk.Label(basic_frame, text="Batch Size:").grid(row=0, column=2, padx=10, sticky=tk.E)
        ttk.Entry(basic_frame, textvariable=self.batch_size, width=10).grid(row=0, column=3, sticky=tk.W, padx=(0,20))
        ttk.Label(basic_frame, text="Learning Rate:").grid(row=0, column=4, padx=10, sticky=tk.E)
        ttk.Entry(basic_frame, textvariable=self.learning_rate, width=12).grid(row=0, column=5, sticky=tk.W)
        row += 1

        # 7. Advanced Hyperparameters
        adv_frame = ttk.LabelFrame(content_frame, text=" Advanced Hyperparameters ", padding=20)
        adv_frame.grid(row=row, column=0, columnspan=4, sticky=tk.W+tk.E, pady=20)
        adv_frame.grid_columnconfigure(0, weight=1); adv_frame.grid_columnconfigure(1, weight=1)

        # Global multiplier
        ttk.Label(adv_frame, text="Global Augmentation Multiplier:", font=('Segoe UI', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=8)
        tk.Scale(adv_frame, from_=1, to=10, resolution=1, variable=self.augment_multiplier,
                 orient=tk.HORIZONTAL, length=400, tickinterval=1).grid(row=0, column=1, columnspan=2, sticky=tk.W+tk.E, padx=10)
        ttk.Label(adv_frame, textvariable=self.augment_multiplier, font=('Segoe UI', 14, 'bold'), foreground='#e74c3c', width=4).grid(row=0, column=3)

        # Per-class augmentation
        per_class_container = ttk.LabelFrame(adv_frame, text=" Per-Class Augmentation Settings (Critical!) ", padding=15)
        per_class_container.grid(row=1, column=0, columnspan=4, sticky=tk.W+tk.E+tk.N+tk.S, pady=15)
        per_class_container.grid_columnconfigure(0, weight=1)
        per_class_container.grid_columnconfigure(1, weight=0)
        per_class_container.grid_rowconfigure(0, weight=1)

        canvas2 = tk.Canvas(per_class_container, height=240, highlightthickness=0)
        v_scroll = ttk.Scrollbar(per_class_container, orient="vertical", command=canvas2.yview)
        self.per_class_frame = ttk.Frame(canvas2)
        self.per_class_frame.bind("<Configure>", lambda e: canvas2.configure(scrollregion=canvas2.bbox("all")))
        canvas2.create_window((0, 0), window=self.per_class_frame, anchor="nw")
        canvas2.configure(yscrollcommand=v_scroll.set)
        canvas2.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N+tk.S)
        v_scroll.grid(row=0, column=1, sticky=tk.N+tk.S)

        # Other advanced parameters
        r = 2
        params = [
            ("Elastic Deform Intensity:", self.intensity_modifier, "scale", 0.05, 1.0, 0.05),
            ("Individual Models Count:", self.num_seeds, "combo", [2,3,4,5,6,7,8,9,10]),
            ("Test Set Ratio:", self.test_size, "scale", 0.10, 0.40, 0.05),
            ("Validation Set Ratio:", self.val_size, "scale", 0.10, 0.30, 0.05),
            ("Early Stopping Patience:", self.early_stop, "entry", None),
            ("Dropout Rate:", self.dropout_rate, "scale", 0.0, 0.8, 0.1),
        ]
        for label_text, var, widget_type, *args in params:
            ttk.Label(adv_frame, text=label_text, font=('Segoe UI', 10)).grid(row=r, column=0, sticky=tk.W, padx=(0,15), pady=8)
            if widget_type == "combo":
                combo = ttk.Combobox(adv_frame, textvariable=var, values=args[0], width=12, state="readonly")
                combo.grid(row=r, column=1, sticky=tk.W, padx=5)
                ttk.Label(adv_frame, textvariable=var, font=('Segoe UI', 12, 'bold'), foreground='#e74c3c', width=6, anchor='center').grid(row=r, column=2, padx=(20,10))
            elif widget_type == "scale":
                tk.Scale(adv_frame, from_=args[0], to=args[1], resolution=args[2], variable=var,
                         orient=tk.HORIZONTAL, length=320, showvalue=False).grid(row=r, column=1, sticky=tk.W+tk.E, padx=5, pady=8)
                ttk.Label(adv_frame, textvariable=var, font=('Segoe UI', 12, 'bold'), foreground='#e74c3c', width=6, anchor='center').grid(row=r, column=2, padx=(20,10))
            elif widget_type == "entry":
                ttk.Entry(adv_frame, textvariable=var, width=12).grid(row=r, column=1, sticky=tk.W, padx=5)
                ttk.Label(adv_frame, textvariable=var, font=('Segoe UI', 12, 'bold'), foreground='#e74c3c', width=6, anchor='center').grid(row=r, column=2, padx=(20,10))
            r += 1
        row += 1

        # # Mode selection
        # mode_frame = ttk.Frame(content_frame)
        # mode_frame.grid(row=row, column=0, columnspan=4, pady=20)
        # ttk.Radiobutton(mode_frame, text="Train + Validate + Test", variable=self.mode, value="train").pack(side=tk.LEFT, padx=20)
        # ttk.Radiobutton(mode_frame, text="Test Only (load existing model)", variable=self.mode, value="test").pack(side=tk.LEFT, padx=20)
        # row += 1

        # Action buttons
        btn_frame = ttk.Frame(content_frame)
        btn_frame.grid(row=row, column=0, columnspan=4, pady=30)
        self.btn1 = ttk.Button(btn_frame, text="Step 1: Convert Excel → Labels", command=self.step1_convert_excel)
        self.btn1.pack(side=tk.LEFT, padx=15)
        self.btn2 = ttk.Button(btn_frame, text="Step 2: BBox Annotation (Optional)", command=self.step2_bbox_annotation)
        self.btn2.pack(side=tk.LEFT, padx=15)
        self.btn3 = ttk.Button(btn_frame, text="Step 3: Start Training and Testing", command=self.step3_train, style='TButton')
        self.btn3.pack(side=tk.LEFT, padx=15)

        # Column weights
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=3)
        content_frame.grid_columnconfigure(2, weight=1)
        content_frame.grid_columnconfigure(3, weight=1)

        # ================== Progress bar and log area fixed to the bottom ==================
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(side=tk.BOTTOM, fill=tk.X, padx=50, pady=(15, 0))

        log_container = ttk.Frame(self.root, relief=tk.RAISED, borderwidth=2)
        log_container.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))

        ttk.Label(log_container, text="GUI Log", font=('Segoe UI', 11, 'bold')).pack(anchor=tk.W, padx=12, pady=(10, 5))

        log_inner = ttk.Frame(log_container)
        log_inner.pack(fill=tk.X, padx=12, pady=(0, 10))

        self.log_text = tk.Text(log_inner, height=11, font=('Consolas', 10), bg='#2c3e50', fg='#ecf0f1',
                                relief=tk.FLAT, padx=12, pady=10, wrap=tk.NONE)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        v_scroll = ttk.Scrollbar(log_inner, command=self.log_text.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=v_scroll.set)

        h_scroll = ttk.Scrollbar(log_container, orient=tk.HORIZONTAL, command=self.log_text.xview)
        h_scroll.pack(fill=tk.X, padx=12, pady=(0, 10))
        self.log_text.config(xscrollcommand=h_scroll.set)

        self.log_message("Welcome to DefectMaster Pro - Edition 2025.12.12")

    def update_per_class_augment(self):
        for widget in self.per_class_frame.winfo_children():
            widget.destroy()
        self.augment_per_class.clear()

        class_names = [c.strip() for c in self.class_names.get().split(",") if c.strip()]
        if not class_names:
            ttk.Label(self.per_class_frame, text="Please enter class names in the box above.", foreground="gray").pack(pady=20)
            return

        headers = ["Class", "Current ×", "Augmentation Multiplier (1-50)", "Recommendation"]
        for col, text in enumerate(headers):
            ttk.Label(self.per_class_frame, text=text, font=('Segoe UI', 10, 'bold'), foreground='#2c3e50').grid(row=0, column=col, sticky=tk.W, padx=8, pady=8)

        for i, name in enumerate(class_names):
            row = i + 1
            ttk.Label(self.per_class_frame, text=f"  {i+1}. {name}", font=('Segoe UI', 10)).grid(row=row, column=0, sticky=tk.W, padx=15, pady=6)

            var = tk.IntVar(value=2 if i > 0 else 1)
            self.augment_per_class[name] = var

            ttk.Label(self.per_class_frame, textvariable=var, font=('Segoe UI', 14, 'bold'),
                      foreground='#e74c3c' if i > 0 else '#27ae60', width=5).grid(row=row, column=1, padx=10)

            scale = tk.Scale(self.per_class_frame, from_=1, to=50, resolution=1, variable=var,
                             orient=tk.HORIZONTAL, length=220, tickinterval=0, showvalue=False)
            scale.grid(row=row, column=2, padx=10, pady=6, sticky=tk.W+tk.E)

            suggest = "Keep 1-5" if i == 0 else "1-30 (minority class higher)"
            color = "#7f8c8d" if i == 0 else "#27ae60"
            ttk.Label(self.per_class_frame, text=suggest, foreground=color,
                      font=('Segoe UI', 9, 'italic'), wraplength=180, justify=tk.LEFT).grid(
                row=row, column=3, padx=15, sticky=tk.W, pady=2)

        tip = ttk.Label(self.per_class_frame,
                        text="Final multiplier = Per-class × Global\nExample: 10 × 3 = 30",
                        foreground="#3498db", font=('Segoe UI', 10, 'bold'), justify=tk.LEFT)
        tip.grid(row=len(class_names)+1, column=0, columnspan=4, pady=(20,10), padx=15, sticky=tk.W)

        self.per_class_frame.grid_columnconfigure(0, weight=1)
        self.per_class_frame.grid_columnconfigure(1, weight=0)
        self.per_class_frame.grid_columnconfigure(2, weight=2)
        self.per_class_frame.grid_columnconfigure(3, weight=1)

    def log_message(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {msg}\n"

        self.log_text.insert(tk.END, formatted_msg)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

        save_dir = self.save_dir.get()
        if save_dir and save_dir.strip():
            log_path = os.path.join(save_dir, "gui_log.txt")
            try:
                os.makedirs(save_dir, exist_ok=True)
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(formatted_msg)
            except:
                pass

    def browse_image_dir(self):
        path = filedialog.askdirectory(title="Select Image Folder")
        if path:
            self.image_dir.set(path)
            imgs = [f for f in os.listdir(path) if
                    f.lower().split('.')[-1] in ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff']]
            self.num_images.set(len(imgs))
            self.log_message(f"Found {len(imgs)} images")

            # recommend experiment save path under current directory
            if not self.save_dir.get().strip():
                exp_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                gui_dir = os.path.dirname(os.path.abspath(__file__))
                auto_path = os.path.join(gui_dir, "experiments", exp_name)
                os.makedirs(auto_path, exist_ok=True)
                self.save_dir.set(auto_path)

                log_path = os.path.join(auto_path, "gui_log.txt")
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(f"GUI Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                self.log_message(f"Auto generated experiment save path: {auto_path}")

    def browse_excel(self):
        path = filedialog.askopenfilename(title="Select Excel Label File", filetypes=[("Excel", "*.xlsx *.xls")])
        if path:
            self.excel_path.set(path)
            self.log_message(f"Excel file selected")

            # generate label file path (same as excel, add _labels)
            npy_dir = os.path.dirname(path)
            npy_name = os.path.splitext(os.path.basename(path))[0] + "_labels.npy"
            auto_npy = os.path.join(npy_dir, npy_name)

            # generate annotation file path (same as excel, add _annotations)
            json_dir = os.path.dirname(path)
            json_name = os.path.splitext(os.path.basename(path))[0] + "_annotations.json"
            auto_json = os.path.join(json_dir, json_name)

            if not self.json_path.get():  # auto fill
                self.npy_path.set(auto_npy)
                self.log_message(f"auto generated label file path：{auto_npy}")
                self.json_path.set(auto_json)
                self.log_message(f"auto generated annotation file path：{auto_json}")

    def browse_json(self):
        path = filedialog.asksaveasfilename(title="Save Annotation File", defaultextension=".json")
        if path: self.json_path.set(path)

    def browse_save_dir(self):
        path = filedialog.askdirectory(title="Select Save Directory")
        if path: self.save_dir.set(path)

    def on_annotate_close(self, window):
        window.destroy()
        self.log_message("Annotation tool closed")

    def check_config(self):
        errors = []
        img_dir = self.image_dir.get()
        excel = self.excel_path.get()
        save_dir = self.save_dir.get()
        class_names = [c.strip() for c in self.class_names.get().split(",") if c.strip()]

        if not img_dir or not os.path.isdir(img_dir):
            errors.append("• Invalid image directory")
        if not excel or not os.path.exists(excel):
            errors.append("• Excel label file not found")
        if len(class_names) < 2:
            errors.append("• At least 2 class names must be defined")
        if not save_dir:
            errors.append("• Experiment save directory not specified")
        else:
            os.makedirs(save_dir, exist_ok=True)

        # Check if npy file exists and is consistent
        if os.path.exists(self.npy_path.get()):
            try:
                labels = np.load(self.npy_path.get())
                if labels.shape[0] != self.num_images.get():
                    errors.append(
                        f"• Image count ({self.num_images.get()}) does not match npy rows ({labels.shape[0]})")
                if labels.shape[1] != len(class_names):
                    errors.append(f"• Class count ({len(class_names)}) does not match npy columns ({labels.shape[1]})")
            except Exception as e:
                errors.append(f"• Corrupted npy file: {e}")

        if errors:
            messagebox.showerror("Configuration Error", "\n".join(errors))
            self.log_message("Configuration check failed!")
        else:
            messagebox.showinfo("Configuration Valid", "All settings are correct. Ready to train!")
            self.log_message("Configuration check passed!")

    def check_thread(self, thread):
        if thread.is_alive():
            self.root.after(100, self.check_thread, thread)
        else:
            self.log_message("Training completed in thread!")
            self.enable_buttons()

    def on_training_completed(self, save_dir):
        self.progress.stop()
        self.enable_buttons()

        self.log_message(f"Training completed! Results saved in: {save_dir}")
        messagebox.showinfo("Success", "Training / Testing completed successfully!")

    def on_training_failed(self, error_traceback):
        self.progress.stop()
        self.enable_buttons()

        messagebox.showerror("Error", error_traceback)
        self.log_message(f"Training failed:\n{error_traceback}")

    def check_training_thread(self):
        if not hasattr(self, 'train_thread') or not self.train_thread.is_alive():
            return

        self.root.after(1000*60*30, self.check_training_thread)

        self.log_message("Training in progress...")

    def disable_buttons(self):
        for btn in [self.btn1, self.btn2, self.btn3]:
            btn.state(['disabled'])

    def enable_buttons(self):
        for btn in [self.btn1, self.btn2, self.btn3]:
            btn.state(['!disabled'])

    def step1_convert_excel(self):
        # Check if all required paths are filled
        if not all([self.image_dir.get(), self.excel_path.get(), self.save_dir.get()]):
            messagebox.showwarning("Warning", "Please select the image folder, Excel file, and save directory first.")
            return

        # Parse class names from the input field
        class_names = [c.strip() for c in self.class_names.get().split(",") if c.strip()]
        if len(class_names) < 2:
            messagebox.showwarning("Warning", "Please define at least 2 class names.")
            return

        # UI feedback: disable buttons and start progress bar
        self.disable_buttons()
        self.progress.start(10)
        self.log_message("Step 1 started: Converting Excel → .npy label file...")

        def convert_thread_func():
            try:
                # Ensure the save directory exists
                os.makedirs(self.save_dir.get(), exist_ok=True)

                # Perform the actual conversion (this is the time-consuming part)
                excel_to_npy_module.excel_to_onehot_npy(
                    columns=class_names,
                    excel_path=self.excel_path.get(),
                    output_npy_path=self.npy_path.get(),
                    num_images=self.num_images.get()
                )

                # On success: show popup and log message in the main thread
                self.root.after(0, lambda: messagebox.showinfo(
                    "Success",
                    f"Step 1 completed successfully!\nLabel file generated:\n{self.npy_path.get()}"
                ))
                self.root.after(0, lambda: self.log_message(
                    f"Step 1 completed: Label file successfully generated → {self.npy_path.get()}"
                ))

            except Exception as e:
                tb = traceback.format_exc()
                self.root.after(0, lambda: messagebox.showerror("Step 1 Failed", str(e)))
                self.root.after(0, lambda: self.log_message(f"Step 1 failed:\n{e}\n{tb}"))

            finally:
                # Always re-enable UI elements when done (or failed)
                self.root.after(0, self.progress.stop)
                self.root.after(0, self.enable_buttons)

        # Run the conversion in a background thread (GUI stays responsive)
        threading.Thread(target=convert_thread_func, daemon=True).start()

    def step2_bbox_annotation(self):
        if not os.path.exists(self.npy_path.get()):
            messagebox.showwarning("Warning", "Please complete Step 1 to generate the label file first")
            return

        class_names = [c.strip() for c in self.class_names.get().split(",") if c.strip()]
        defect_types_for_bbox = {
            i: name for i, name in enumerate(class_names)
        }

        if not defect_types_for_bbox:
            messagebox.showinfo("Info", "No defect classes detected. Skipping annotation (you may close this window)")

        annotation_root = tk.Toplevel(self.root)
        annotation_root.title(f"Bounding Box Annotation Tool - {len(defect_types_for_bbox)} Classes")
        annotation_root.geometry("1100x750")
        annotation_root.grab_set()

        # Dynamic injection
        app = bbox_tool_module.BBoxAnnotationTool(
            annotation_root,
            image_dir=self.image_dir.get(),
            label_file=self.npy_path.get(),
            output_file=self.json_path.get(),
            defect_types=defect_types_for_bbox
        )

        self.log_message(f"Annotation tool launched successfully - {len(defect_types_for_bbox)} classes available")

        def on_close():
            app.save_annotations()
            self.log_message("Annotations saved. Window closed.")
            annotation_root.destroy()

        annotation_root.protocol("WM_DELETE_WINDOW", on_close)

        # Start the annotation tool
        app.run()


    def step3_train(self):
        class_names = [c.strip() for c in self.class_names.get().split(",") if c.strip()]
        self.log_message(f"class_names: {class_names}")
        if len(class_names) < 2:
            messagebox.showerror("Error", "At least 2 classes must be defined!")
            return

        # Collect per-class augmentation multipliers
        augment_counts = {name: var.get() for name, var in self.augment_per_class.items()}
        self.log_message(f"augment_counts: {augment_counts}")
        augment_counts['default'] = 1

        self.disable_buttons()
        self.progress.start(10)

        try:
            config = train_module.Config(
                image_dir=self.image_dir.get(),
                label_file=self.npy_path.get(),
                annotation_file=self.json_path.get() or None,
                save_root_dir=self.save_dir.get(),
                num_epochs=self.num_epochs.get(),
                batch_size=self.batch_size.get(),
                learning_rate=self.learning_rate.get(),
                sample_num=self.num_images.get(),
                num_classes=len(class_names),
                class_names=class_names,
                class_names_cn=class_names,
                test_size=self.test_size.get(),
                val_size=self.val_size.get(),
                early_stop_patience=self.early_stop.get(),
                dropout_rate=self.dropout_rate.get(),
                random_seeds=random.sample(range(10000), 10)[:self.num_seeds.get()],
                augment_counts_expand=self.augment_multiplier.get(),
                augment_counts=augment_counts,
                intensity_modifier=self.intensity_modifier.get(),
            )

            os.makedirs(config.save_root_dir, exist_ok=True)

            self.log_message("Training Configuration:")
            config_dict = config.save_config()
            for key, value in config_dict.items():
                self.log_message(f"  {key}: {value}")

        except Exception as e:
            tb = traceback.format_exc()
            messagebox.showerror("Config Error", tb)
            self.log_message(f"Config creation failed:\n{tb}")
            self.progress.stop()
            self.enable_buttons()
            return

        def train_thread_func():
            try:
                # if self.mode.get() == "train":
                #     self.log_message(f"mode: train_validate_test")
                #     train_module.train_validate_test(config)
                # else:
                #     self.log_message(f"mode: test")
                #     train_module.test_only(config)
                self.log_message(f"mode: train_validate_test")
                train_module.train_validate_test(config)

                self.root.after(0, self.on_training_completed, config.save_root_dir)

            except Exception as e:
                tb = traceback.format_exc()
                self.root.after(0, self.on_training_failed, tb)

        self.train_thread = threading.Thread(target=train_thread_func, daemon=True)
        self.train_thread.start()

        self.root.after(100, self.check_training_thread)


if __name__ == "__main__":
    root = tk.Tk()
    app = DefectTrainingPro(root)
    root.mainloop()
