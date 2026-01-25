import tkinter as tk
from PIL import Image, ImageTk
import json
from pathlib import Path
import numpy as np


class BBoxAnnotationTool:
    def __init__(self, root, image_dir, label_file, output_file, defect_types=None):
        self.root = root
        self.root.title("Bounding Box Annotation Tool")

        self.image_dir = Path(image_dir)
        self.labels = np.load(label_file)
        self.output_file = output_file
        self.image_paths = [self.image_dir / f"{i + 1}.png" for i in range(self.labels.shape[0])]
        self.current_idx = 0
        self.scale_x = 1.0
        self.scale_y = 1.0

        if defect_types is not None:
            self.defect_types = defect_types
        else:
            self.defect_types = {0: 'no_defect', 1: 'defect1', 2: 'defect2', 3: 'defect3'}

        for img_path in self.image_paths:
            if not img_path.exists():
                raise ValueError(f"Img {img_path} does not exist!")

        self.annotations = {}
        if Path(self.output_file).exists():
            try:
                with open(self.output_file, 'r') as f:
                    self.annotations = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Loading annotation file {self.output_file} failed.") from e

        self.info_label = tk.Label(root, text="", font=("Arial", 12), anchor="w", justify="left", bg="white",
                                   fg="black")
        self.info_label.pack(fill="x", padx=10, pady=5)

        # GUI
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        self.label_frame = tk.Frame(root)
        self.label_frame.pack()
        self.defect_var = tk.StringVar(value='dislocation')
        for idx, name in self.defect_types.items():
            if idx == 0: continue  # skip no defect
            tk.Radiobutton(
                self.label_frame, text=name, variable=self.defect_var, value=name
            ).pack(side=tk.LEFT)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()
        tk.Button(self.button_frame, text="Previous", command=self.prev_image).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Next", command=self.next_image).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Save", command=self.save_annotations).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Clear Current Box", command=self.clear_bbox).pack(side=tk.LEFT)

        # Status
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.image = None
        self.photo = None
        self.bboxes = []

        # Mouse
        self.canvas.bind("<Button-1>", self.start_rect)
        self.canvas.bind("<B1-Motion>", self.update_rect)
        self.canvas.bind("<ButtonRelease-1>", self.end_rect)

        # load the first image
        self.load_image()

    def load_image(self):
        if not (0 <= self.current_idx < len(self.image_paths)):
            self.canvas.create_text(400, 300, text="No more images.", font=("Arial", 16))
            return

        img_path = self.image_paths[self.current_idx]
        if not img_path.exists():
            self.canvas.create_text(400, 300, text=f"Img {img_path} does not exist!", font=("Arial", 16))
            return

        try:
            self.image = Image.open(img_path)
            original_size = self.image.size
            max_size = (800, 600)
            self.image.thumbnail(max_size, Image.Resampling.LANCZOS)
            display_size = self.image.size
            self.scale_x = original_size[0] / display_size[0]
            self.scale_y = original_size[1] / display_size[1]

            # clear old PhotoImage
            if hasattr(self.canvas, 'image'):
                self.canvas.image = None

            # create new PhotoImage
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.image = self.photo
            self.canvas.config(width=display_size[0], height=display_size[1])
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # show class
            label = self.labels[self.current_idx]
            defect_names = [self.defect_types[i] for i in np.where(label == 1)[0]]
            self.info_label.config(text=f"Img {self.current_idx + 1}: {', '.join(defect_names)}")

            # load annotation
            self.bboxes = []
            for bbox in self.annotations.get(str(self.current_idx + 1), []):
                x_min, y_min, x_max, y_max = bbox['bbox']
                self.bboxes.append({
                    'defect_type': bbox['defect_type'],
                    'bbox': [x_min / self.scale_x, y_min / self.scale_y, x_max / self.scale_x, y_max / self.scale_y]
                })
            self.redraw_bboxes()
        except Exception as e:
            self.canvas.create_text(400, 300, text=f"Failed to load image: {str(e)}", font=("Arial", 16))

    def start_rect(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2
        )

    def update_rect(self, event):
        if self.current_rect:
            curr_x = self.canvas.canvasx(event.x)
            curr_y = self.canvas.canvasy(event.y)
            self.canvas.coords(self.current_rect, self.start_x, self.start_y, curr_x, curr_y)

    def end_rect(self, event):
        if self.current_rect:
            curr_x = self.canvas.canvasx(event.x)
            curr_y = self.canvas.canvasy(event.y)
            x_min, y_min = min(self.start_x, curr_x), min(self.start_y, curr_y)
            x_max, y_max = max(self.start_x, curr_x), max(self.start_y, curr_y)
            defect_type = self.defect_var.get()
            self.bboxes.append({
                'defect_type': defect_type,
                'bbox': [x_min, y_min, x_max, y_max]
            })
            self.canvas.delete(self.current_rect)
            self.current_rect = None
            self.redraw_bboxes()

    def redraw_bboxes(self):
        self.canvas.delete("bbox")
        for bbox in self.bboxes:
            x_min, y_min, x_max, y_max = bbox['bbox']
            self.canvas.create_rectangle(
                x_min, y_min, x_max, y_max, outline='blue', width=2, tags="bbox"
            )
            self.canvas.create_rectangle(
                x_min - 5, y_min, x_max + 5, y_min + 20,
                fill='black', outline='', tags="bbox", stipple="gray50"
            )
            self.canvas.create_text(
                x_min + 30, y_min + 8, text=bbox['defect_type'], fill='yellow', tags="bbox", font=("Arial", 12, 'bold')
            )

    def clear_bbox(self):
        self.bboxes = []
        self.redraw_bboxes()

    def prev_image(self):
        if self.current_idx > 0:
            self.save_current_bboxes()
            self.current_idx -= 1
            self.load_image()

    def next_image(self):
        if self.current_idx < len(self.image_paths) - 1:
            self.save_current_bboxes()
            self.current_idx += 1
            self.load_image()

    def save_current_bboxes(self):
        scaled_bboxes = [
            {
                'defect_type': bbox['defect_type'],
                'bbox': [
                    bbox['bbox'][0] * self.scale_x, bbox['bbox'][1] * self.scale_y,
                    bbox['bbox'][2] * self.scale_x, bbox['bbox'][3] * self.scale_y
                ]
            }
            for bbox in self.bboxes
        ]
        self.annotations[str(self.current_idx + 1)] = scaled_bboxes

    def save_annotations(self):
        self.save_current_bboxes()
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f, indent=4)
        print(f"Annotations have been saved to {self.output_file}")

    def run(self):
        self.root.mainloop()