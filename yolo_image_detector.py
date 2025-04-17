import os
# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tkinter as tk
from tkinter import Canvas, Label, Frame, scrolledtext, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os
import sqlite3
import json
import re
import faiss
import torch
import open_clip

# Add PyTorch 2.6+ compatibility fix
try:
    # Try to import the required function to allow YOLOv8 model classes
    from torch.serialization import add_safe_globals
    # Add the ultralytics detection model to the allowlist
    add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
    print("Added YOLOv8 model class to PyTorch safe globals list")
except ImportError:
    # Older PyTorch versions don't have this function
    print("Using older PyTorch version or couldn't import safe globals function")

# Database path
DB_PATH = "data/pokemon_cards_simple.db"

# Path translation constants
DB_IMAGE_PATH = "data/assets/images"
FILESYSTEM_IMAGE_PATH = "data/assets/cards"

# CLIP model paths
FAISS_INDEX_PATH = "data/clip/card_vectors_clip_vitb32.faiss"
CARD_IDS_PATH = "data/clip/card_ids_clip_vitb32.json"

# Import TkinterDnD2
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
except ImportError:
    print("tkinterdnd2 not installed. Please install it with 'pip install tkinterdnd2'")
    raise

# Import PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    print("PaddleOCR not installed. Install with 'pip install paddleocr'")
    PADDLE_AVAILABLE = False

# Import fuzzy matching
try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    print("Warning: fuzzywuzzy not installed. Using exact matching only.")
    print("Install with: pip install fuzzywuzzy[speedup]")
    FUZZY_AVAILABLE = False

class YoloImageDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Pokemon Card Detector with OCR")
        self.root.geometry("1900x900") # Increased width for 4 images
        
        # Load the YOLO models
        self.detector_model_path = r"models/detector/best.pt"
        self.model_path = r"models/classifier/best.pt"
        
        # Load both models with compatibility handling for PyTorch 2.6+
        try:
            # First try with safe globals approach (should work if add_safe_globals succeeded)
            self.detector_model = YOLO(self.detector_model_path)
            self.model = YOLO(self.model_path)
            print("Successfully loaded models with safe globals")
        except Exception as e:
            print(f"Failed to load models with default settings: {e}")
            print("Trying with weights_only=False for PyTorch 2.6+ compatibility...")
            
            # Patch the torch.load function to use weights_only=False
            original_load = torch.load
            def patched_load(file, *args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(file, *args, **kwargs)
            
            # Apply the patch
            torch.load = patched_load
            
            try:
                # Try loading again with the patch
                self.detector_model = YOLO(self.detector_model_path)
                self.model = YOLO(self.model_path)
                print("Successfully loaded models with weights_only=False")
            finally:
                # Restore original function regardless of success
                torch.load = original_load
        
        # Get class names from the model
        self.class_names = self.model.names
        
        # Add status variable
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        # Initialize PaddleOCR if available
        if PADDLE_AVAILABLE:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            
        # Initialize CLIP model and resources
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_index = None
        self.clip_card_ids = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_clip_resources()
        
        # Create the main frame
        self.main_frame = Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create instructional label
        self.instruction_label = Label(
            self.main_frame, 
            text="Drag and drop a Pokemon card image to detect regions and extract text",
            font=("Arial", 14)
        )
        self.instruction_label.pack(pady=10)
        
        # Create upper frame with input image, clip match, db match, and combined match
        upper_frame = Frame(self.main_frame)
        upper_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Left side: Input image canvas
        input_frame = Frame(upper_frame)
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        Label(input_frame, text="Input Image:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.input_canvas = Canvas(input_frame, bg="lightgray")
        self.input_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Middle-Left: CLIP Matched card image
        clip_match_frame = Frame(upper_frame)
        clip_match_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        Label(clip_match_frame, text="CLIP Match:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.clip_match_canvas = Canvas(clip_match_frame, bg="lightgray")
        self.clip_match_canvas.pack(fill=tk.BOTH, expand=True)

        # Middle-Right: DB Matched card image
        db_match_frame = Frame(upper_frame)
        db_match_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        Label(db_match_frame, text="Database Match:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.db_match_canvas = Canvas(db_match_frame, bg="lightgray")
        self.db_match_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right side: Combined Match card image
        combined_match_frame = Frame(upper_frame)
        combined_match_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        Label(combined_match_frame, text="Combined Match:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.combined_match_canvas = Canvas(combined_match_frame, bg="lightgray")
        self.combined_match_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create info frame for displaying detection and OCR results
        self.info_frame = Frame(self.root)
        self.info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create a frame for detection results
        detection_frame = Frame(self.info_frame)
        detection_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        Label(detection_frame, text="Detection Results:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        self.results_label = Label(
            detection_frame,
            text="Drop an image to see detection results",
            font=("Arial", 12),
            justify=tk.LEFT
        )
        self.results_label.pack(anchor=tk.W, fill=tk.X)
        
        # Create a frame for OCR results
        ocr_frame = Frame(self.info_frame)
        ocr_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        Label(ocr_frame, text="OCR Results:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        self.ocr_text = scrolledtext.ScrolledText(
            ocr_frame, 
            height=10,
            width=50,
            font=("Arial", 11)
        )
        self.ocr_text.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for database search results
        search_frame = Frame(self.root)
        search_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        Label(search_frame, text="Card Database Matches:", 
              font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        # Create a treeview for displaying search results
        self.results_tree = ttk.Treeview(search_frame, columns=("name", "set", "number", "attacks", "relevance"))
        self.results_tree.heading("#0", text="ID")
        self.results_tree.heading("name", text="Name")
        self.results_tree.heading("set", text="Set")
        self.results_tree.heading("number", text="Number")
        self.results_tree.heading("attacks", text="Attacks")
        self.results_tree.heading("relevance", text="Relevance")
        
        self.results_tree.column("#0", width=120)
        self.results_tree.column("name", width=150) 
        self.results_tree.column("set", width=100)
        self.results_tree.column("number", width=80)
        self.results_tree.column("attacks", width=300)
        self.results_tree.column("relevance", width=80)
        
        # Add a scrollbar to the treeview
        tree_scroll = ttk.Scrollbar(search_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=tree_scroll.set)
        
        # Pack the treeview and scrollbar
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event to show the matched card (DB match)
        self.results_tree.bind("<<TreeviewSelect>>", self.on_db_result_select) # Renamed handler
        
        # Setup drag and drop bindings (bind to input_canvas)
        self.input_canvas.drop_target_register(DND_FILES)
        self.input_canvas.dnd_bind('<<Drop>>', self.on_drop)
        
        # Variables to store image data
        self.original_image = None
        self.input_photo = None 
        self.clip_match_photo = None 
        self.db_match_photo = None 
        self.combined_match_photo = None # New
        
        # OCR result variables
        self.detected_name = None
        self.detected_attacks = []
        self.detected_number = None
        
        # CLIP similarity variables
        self.current_input_image_path = None
        self.best_clip_match_info = None
        self.input_image_embedding = None # Store embedding of the input image
        # Store DB search results
        self.search_results = []
        
    def load_clip_resources(self):
        """Load CLIP model, FAISS index, and card metadata for image similarity"""
        try:
            print("Loading CLIP (ViT-B/32) model...")
            print("  Downloading/loading CLIP model weights (this may take a while)...", flush=True)
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32',
                pretrained='laion2b_s34b_b79k'
            )
            self.clip_model.to(self.device)
            self.clip_model.eval()
            print(f"CLIP model loaded successfully onto {self.device}.")

            print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
            if not os.path.exists(FAISS_INDEX_PATH):
                print(f"Warning: FAISS index file not found: {FAISS_INDEX_PATH}")
                return
            self.clip_index = faiss.read_index(FAISS_INDEX_PATH)
            print(f"FAISS index loaded ({self.clip_index.ntotal} vectors).")

            print(f"Loading card metadata from {CARD_IDS_PATH}...")
            if not os.path.exists(CARD_IDS_PATH):
                print(f"Warning: Card ID file not found: {CARD_IDS_PATH}")
                return
            with open(CARD_IDS_PATH, "r") as f:
                self.clip_card_ids = json.load(f)
                # Convert keys back to integers if they were saved as strings
                self.clip_card_ids = {int(k): v for k, v in self.clip_card_ids.items()}
            print("Card metadata loaded.")
            
        except Exception as e:
            print(f"Error loading CLIP resources: {e}")
            
    def preprocess_image_for_clip(self, image_path):
        """Load and preprocess image using CLIP's preprocessor"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Use CLIP's preprocessor
            img_tensor = self.clip_preprocess(img).unsqueeze(0)  # Add batch dim
            return img_tensor
        except Exception as e:
            print(f"Error preprocessing image {image_path} for CLIP: {e}")
            return None
            
    def get_clip_embedding(self, image_path):
        """Generate a normalized CLIP embedding for a given image path."""
        if not self.clip_model or not self.clip_preprocess:
            print("CLIP model/preprocessor not loaded.")
            return None

        # Preprocess image
        img_tensor = self.preprocess_image_for_clip(image_path)
        if img_tensor is None:
            return None

        # Get embedding
        try:
            with torch.no_grad():
                image_tensor = img_tensor.to(self.device)
                embedding_tensor = self.clip_model.encode_image(image_tensor)
                # Normalize
                embedding_tensor /= embedding_tensor.norm(dim=-1, keepdim=True)
                # Move to CPU and convert to NumPy array
                embedding = embedding_tensor.cpu().numpy()[0]
            return embedding
        except Exception as e:
            print(f"Failed to generate CLIP embedding for {image_path}: {e}")
            return None

    def find_closest_card_with_clip(self, image_path, k=3):
        """Find the top k closest matching cards using CLIP embeddings and FAISS."""
        if not self.clip_index or not self.clip_card_ids:
            print("CLIP index/metadata not loaded. Skipping FAISS search.")
            return []

        print(f"Processing query image with CLIP for FAISS search: {image_path}")
        # Get embedding for the query image
        query_embedding = self.get_clip_embedding(image_path)
        if query_embedding is None:
            print(f"Could not get CLIP embedding for query image: {image_path}")
            # Store None for the input embedding if it fails here
            self.input_image_embedding = None
            return []

        # Store the input image embedding for later use
        self.input_image_embedding = query_embedding

        # Prepare for FAISS search
        embedding_matrix = np.array([query_embedding]).astype("float32")

        print(f"\nSearching FAISS index for {k} nearest neighbors (L2 Distance)...")
        # Distances are L2 distances, lower is better
        distances, indices = self.clip_index.search(embedding_matrix, k)

        print("\nCLIP search results:")
        matches = []
        for i in range(k):
            if i >= len(indices[0]) or indices[0][i] == -1:
                print(f"Match {i+1}: Invalid index found.")
                break
            nearest_index = indices[0][i]
            distance = distances[0][i]

            print(f"\nMatch {i+1}:")
            print(f"  Index: {nearest_index}")
            print(f"  Distance (L2): {distance:.4f}")

            if nearest_index in self.clip_card_ids:
                match_info = self.clip_card_ids[nearest_index]
                matches.append((match_info, distance))
                print(f"  Card: {match_info.get('set', '?')}-{match_info.get('number', '?')}")
                print(f"  Filename: {match_info.get('filename', '?')}")
            else:
                print(f"  Warning: Index {nearest_index} not found in metadata")
        return matches

    def on_drop(self, event):
        """Handle file drop event"""
        # Get the file path from the event
        file_path = event.data
        
        # Remove curly braces if present (Windows drag and drop)
        if file_path.startswith("{") and file_path.endswith("}"):
            file_path = file_path[1:-1]
            
        # Store the current input image path for CLIP processing
        self.current_input_image_path = file_path
        
        # Clear previous results/images
        self.input_canvas.delete("all")
        self.clip_match_canvas.delete("all")
        self.db_match_canvas.delete("all")
        self.combined_match_canvas.delete("all") # Clear new canvas
        self.results_tree.delete(*self.results_tree.get_children())
        self.ocr_text.delete(1.0, tk.END)
        self.results_label.config(text="Processing...")
        self.detected_name = None
        self.detected_attacks = []
        self.detected_number = None
        self.original_image = None 
        self.best_clip_match_info = None 
        self.search_results = [] # Clear DB results
        self.input_image_embedding = None # Reset input embedding
        
        # Load and display the input image
        try:
            # Load the image using PIL
            self.original_image = Image.open(file_path).convert("RGB")
            
            # Display original image on the input canvas with "Original Image" label
            self.display_image_on_canvas(self.original_image, self.input_canvas, 'input_photo', "Original Image")

            # First detect and crop using the detector model
            img_cv = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            detector_results = self.detector_model(img_cv, conf=0.25)  # Use 0.25 confidence threshold
            
            if detector_results and len(detector_results[0].boxes) > 0:
                # Process each detected card
                for i, box in enumerate(detector_results[0].boxes):
                    # Get coordinates and crop
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    
                    # Add padding to the crop
                    padding = 10
                    h, w = img_cv.shape[:2]
                    crop_x1 = max(0, x1 - padding)
                    crop_y1 = max(0, y1 - padding)
                    crop_x2 = min(w, x2 + padding)
                    crop_y2 = min(h, y2 + padding)
                    
                    crop = img_cv[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    # Convert crop to PIL Image for processing
                    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    
                    # Update input canvas to show the cropped version
                    if i == 0:  # Show first crop in input canvas
                        self.display_image_on_canvas(crop_pil, self.input_canvas, 'input_photo', f"Detected Card {i+1} (Conf: {conf:.2f})")
                    
                    # Save crop temporarily
                    temp_crop_path = f"temp_crop_{i}.jpg"
                    crop_pil.save(temp_crop_path)
                    
                    # Process the crop through the existing workflow
                    self._process_crop(temp_crop_path, conf)
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_crop_path)
                    except:
                        pass
                
                self.results_label.config(text=f"Processed {len(detector_results[0].boxes)} detected cards")
            else:
                self.results_label.config(text="No cards detected in the image")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or process image: {str(e)}")
            self.results_label.config(text="Error loading image.")

    def _process_crop(self, crop_path, detection_conf):
        """Process a single cropped card image"""
        try:
            # --- Start processing ---
            # 1. Run CLIP similarity check
            self.current_input_image_path = crop_path
            self.run_clip_similarity_check()

            # 2. Run YOLO detection and OCR
            self.detect_objects_and_run_ocr()
            
            # 3. Determine and display the combined match
            self.update_combined_match()

        except Exception as e:
            self.status_var.set(f"Error processing crop: {str(e)}")

    def display_image_on_canvas(self, img_pil, canvas, photo_attr_name, title=None):
        """Resizes and displays a PIL image on a specified canvas."""
        canvas.delete("all") # Clear previous image

        # Calculate canvas dimensions
        canvas.update_idletasks() # Ensure dimensions are up-to-date
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        # Default size if canvas not rendered yet
        if canvas_width <= 1: canvas_width = 400 # Adjust default as needed
        if canvas_height <= 1: canvas_height = 600

        # Display title if provided
        if title:
            canvas.create_text(10, 10, text=title, anchor=tk.NW, font=("Arial", 10, "bold"))
            title_height = 30  # Approximate height for title
        else:
            title_height = 0

        # Calculate scaling to fit the canvas (accounting for title if present)
        img_w, img_h = img_pil.size
        if img_w == 0 or img_h == 0: return # Skip if image is invalid

        available_height = canvas_height - title_height
        scale_w = canvas_width / img_w
        scale_h = available_height / img_h
        scale = min(scale_w, scale_h, 1.0) # Don't scale up

        # Resize the image
        new_width = int(img_w * scale)
        new_height = int(img_h * scale)
        
        # Prevent zero dimensions
        if new_width <= 0: new_width = 1
        if new_height <= 0: new_height = 1

        resized_img = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to PhotoImage and store it as an instance attribute to prevent GC
        photo_image = ImageTk.PhotoImage(resized_img)
        setattr(self, photo_attr_name, photo_image) # Store e.g., self.input_photo

        # Display the image below the title
        canvas.create_image(
            canvas_width / 2, title_height + (available_height / 2), # Center image below title
            image=photo_image, anchor=tk.CENTER
        )
            
    def run_clip_similarity_check(self):
        """Run CLIP FAISS search, display best match, store its info and input embedding."""
        self.clip_match_canvas.delete("all") # Clear previous match
        self.best_clip_match_info = None # Reset before running
        self.input_image_embedding = None # Reset input embedding

        if not self.current_input_image_path:
            return

        # Find closest matching card(s) using CLIP + FAISS
        # This call will set self.input_image_embedding if successful
        matches = self.find_closest_card_with_clip(self.current_input_image_path, k=1)

        if not matches:
            print("No CLIP matches found via FAISS.")
            # Even if no match, find_closest_card_with_clip might have set the input embedding
            if self.input_image_embedding is None:
                print("Could not get input image embedding.")
            self.clip_match_canvas.create_text(10, 10, text="No CLIP match found.", anchor=tk.NW, fill="red")
            return

        # Store and Display the top match from FAISS
        self.best_clip_match_info, distance = matches[0]
        similarity = max(0, min(100, (1 - distance/2.0) * 100))
        print(f"Top CLIP Match (FAISS): {self.best_clip_match_info.get('name', '?')} (Similarity: {similarity:.1f}%)")
        self.display_card_match(self.best_clip_match_info, self.clip_match_canvas, 'clip_match_photo', f"CLIP Match (Sim: {similarity:.1f}%)")

        # Input embedding should now be stored in self.input_image_embedding by find_closest_card_with_clip
        if self.input_image_embedding is None:
             print("Warning: Input image embedding was not set after CLIP search.")

    def detect_objects_and_run_ocr(self):
        """Run YOLO detection, OCR, DB search, display top DB match, and store DB results."""
        if self.original_image is None:
            self.results_label.config(text="No image loaded.")
            return

        # Convert PIL Image to OpenCV format for processing
        img_cv = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)

        # Run YOLO detection
        try:
            results = self.model(img_cv)
            if not results:
                 self.results_label.config(text="Detection failed.")
                 return 
            yolo_result = results[0] # Get the first result object
        except Exception as e:
            self.results_label.config(text=f"Detection Error: {e}")
            return

        # Clear previous OCR/DB results (Treeview cleared in on_drop)
        self.ocr_text.delete(1.0, tk.END)
        self.detected_name = None
        self.detected_attacks = []
        self.detected_number = None
        detected_regions_summary = []

        # Process detection results (for summary label)
        boxes = yolo_result.boxes
        if len(boxes) > 0:
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                detected_regions_summary.append(f"{class_name} ({conf:.2f})")
        
        # Update detection results label
        if detected_regions_summary:
            self.results_label.config(text="Detected: " + ", ".join(detected_regions_summary))
        else:
            self.results_label.config(text="No objects detected by YOLO")

        # Run OCR on detected regions if available and boxes were found
        if PADDLE_AVAILABLE and len(boxes) > 0:
            self.extract_text_from_regions(img_cv, yolo_result) 
            
            # Now that OCR is done, search the database if we have text
            if self.detected_name or self.detected_attacks or self.detected_number:
                 # Store the results from the database search
                 self.search_results = self.search_card_database() 
                 
                 # Display top DB match if found
                 if self.search_results:
                     top_db_match = self.search_results[0]['card'] 
                     relevance = self.search_results[0]['relevance']
                     print(f"Top DB Match: {top_db_match.get('name', '?')} (Relevance: {relevance:.1f}%)")
                     # Display on the dedicated DB match canvas
                     self.display_card_match(top_db_match, self.db_match_canvas, 'db_match_photo', f"DB Match (Rel: {relevance:.1f}%)")
                 else:
                     # No DB matches found, clear the DB canvas
                     self.db_match_canvas.delete("all")
                     self.db_match_canvas.create_text(10, 10, text="No DB match found.", anchor=tk.NW, fill="red")
            else:
                 # No text extracted for search, clear DB canvas
                 self.db_match_canvas.delete("all")
                 self.db_match_canvas.create_text(10, 10, text="No text for DB search.", anchor=tk.NW, fill="orange")
                 self.ocr_text.insert(tk.END, "No text extracted for DB search.\n")

        elif not PADDLE_AVAILABLE:
             self.ocr_text.insert(tk.END, "PaddleOCR not available. Skipping text extraction.\n")
             self.db_match_canvas.delete("all")
             self.db_match_canvas.create_text(10, 10, text="OCR Unavailable", anchor=tk.NW, fill="orange")
        else: # No boxes found
             self.ocr_text.insert(tk.END, "No regions detected for OCR.\n")
             self.db_match_canvas.delete("all")
             self.db_match_canvas.create_text(10, 10, text="No regions detected", anchor=tk.NW, fill="orange")

    def extract_text_from_regions(self, img, result):
        """Extract text using OCR from regions detected by YOLO."""
        height, width = img.shape[:2]
        boxes = result.boxes
        class_names = result.names # Get class names from the result object

        # Clear previous OCR results (already done in calling method)
        # self.ocr_text.delete(1.0, tk.END)
        # self.detected_name = None
        # self.detected_attacks = []
        # self.detected_number = None
        
        if len(boxes) == 0:
            self.ocr_text.insert(tk.END, "No regions detected for OCR.\n")
            return
        
        self.ocr_text.insert(tk.END, "Extracting text from detected regions...\n\n")
        
        # Process each detected box
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0].item())
            class_name = class_names[class_id] # Use names from result object
            # confidence = box.conf[0].item() # Confidence not needed here
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width-1, x2), min(height-1, y2)
            
            # Extract region
            region = img[y1:y2, x1:x2]
            
            # Skip if region is empty
            if region.size == 0:
                self.ocr_text.insert(tk.END, f"{class_name} region: Skipped (empty).\n\n")
                continue
            
            # Run OCR on the region
            try:
                # Use PaddleOCR instance
                ocr_result = self.ocr.ocr(region, cls=True) 
                
                # Extract and format OCR results
                text_result = f"{class_name} region:"
                extracted_text = ""
                
                # Check if ocr_result is structured as expected
                if ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0 and ocr_result[0] is not None:
                    lines = ocr_result[0]
                    if isinstance(lines, list):
                         concatenated_text = ""
                         confidences = []
                         for line in lines:
                            if isinstance(line, list) and len(line) == 2:
                                text_info = line[1]
                                if isinstance(text_info, (tuple, list)) and len(text_info) == 2:
                                    text, confidence = text_info
                                    text_result += f"\\n   • {text} ({confidence:.2f})"
                                    concatenated_text += text + " "
                                    confidences.append(confidence)
                                else:
                                     # Handle cases where the structure might be different
                                     text_result += f"\\n   • Unexpected line format: {line}"
                            else:
                                text_result += f"\\n   • Unexpected item format: {line}"
                        
                         # Use the concatenated text, maybe average confidence if needed
                         if concatenated_text:
                              extracted_text = concatenated_text.strip()

                    else:
                         text_result += "\\n   • Unexpected lines format in result."

                else:
                    text_result += "\\n   • No text detected or unexpected result format"
                
                self.ocr_text.insert(tk.END, text_result + "\n\n")
                
                # Store the detected text based on class
                if extracted_text:
                    # Simple assignment based on class name
                    if class_name == "name":
                        self.detected_name = extracted_text
                    elif class_name == "attack":
                         # If multiple attack boxes, append
                        self.detected_attacks.append(extracted_text) 
                    elif class_name == "number":
                        # Try to extract just the first number from a format like "xxx/yyy"
                        number_match = re.search(r'(\d+)\s*\/?\s*\d*', extracted_text) # Made regex more flexible
                        if number_match:
                            self.detected_number = number_match.group(1)
                        else:
                            # Fallback to the raw text if no pattern match
                             self.detected_number = extracted_text.strip() 
                
            except Exception as e:
                self.ocr_text.insert(tk.END, f"{class_name} region: Error performing OCR: {str(e)}\n\n")
        
        # Show a summary of what we'll search for
        self.ocr_text.insert(tk.END, "\n--- Search Parameters ---\n")
        if self.detected_name:
            self.ocr_text.insert(tk.END, f"Name: {self.detected_name}\n")
        if self.detected_attacks:
            # Join attacks if list is not empty
            attacks_str = ', '.join(self.detected_attacks) if self.detected_attacks else "None"
            self.ocr_text.insert(tk.END, f"Attacks: {attacks_str}\n")
        if self.detected_number:
            self.ocr_text.insert(tk.END, f"Number: {self.detected_number}\n")
        if not self.detected_name and not self.detected_attacks and not self.detected_number:
             self.ocr_text.insert(tk.END, "No text extracted for search.\n")

    def search_card_database(self, threshold=60):
        """Search the database using OCR results and populate the Treeview."""
        # Clear previous DB results Treeview
        self.results_tree.delete(*self.results_tree.get_children()) 
        
        self.search_results = self.search_pokemon_by_name_and_attacks( # This func performs the actual DB query
            name=self.detected_name,
            attacks=self.detected_attacks,
            number=self.detected_number,
            threshold=threshold
        )
        
        if not self.search_results:
            self.ocr_text.insert(tk.END, "\nNo matching cards found in database.\n")
            return [] # Return empty list if no matches
        
        # Display results in the tree
        self.ocr_text.insert(tk.END, f"\nFound {len(self.search_results)} potential DB matches.\n")
        
        for i, match in enumerate(self.search_results):
            card = match['card']
            relevance = match['relevance']
            
            # Format attacks for display
            attack_list = "None"
            try:
                attacks_json = json.loads(card.get('attacks', '[]'))
                if attacks_json:
                    attack_names = [a.get('name', 'Unknown') for a in attacks_json]
                    attack_list = ", ".join(attack_names)
            except (json.JSONDecodeError, AttributeError):
                pass
            
            # Insert into tree
            card_id = card.get('id', f"db_unknown_{i}") # Ensure unique ID
            self.results_tree.insert(
                "", "end", text=card_id, # Use card ID as text
                values=(
                    card.get('name', 'Unknown'),
                    card.get('set_name', 'Unknown'),
                    card.get('number', 'Unknown'),
                    attack_list,
                    f"{relevance:.1f}%"
                )
            )
            
        # Return the list of matches (useful for the caller)
        return self.search_results

    def search_pokemon_by_name_and_attacks(self, name=None, attacks=None, number=None, threshold=60):
        """
        Search for Pokemon cards by name, attacks, and optionally number
        using fuzzy matching to handle OCR imperfections.
        
        Args:
            name (str): Pokemon name to search for
            attacks (list): List of attack names to search for
            number (str): Card number to filter by
            threshold (int): Similarity threshold (0-100) for fuzzy matching
        
        Returns:
            list: Matching card records sorted by relevance
        """
        if not name and not attacks:
            self.ocr_text.insert(tk.END, "\nError: Must provide at least a name or attacks to search\n")
            return []
        
        # Connect to database
        if not os.path.exists(DB_PATH):
            self.ocr_text.insert(tk.END, f"\nError: Cannot find database at '{DB_PATH}'\n")
            return []
        
        conn = None
        matches = []
        
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get all potential cards
            cursor.execute("SELECT * FROM cards")
            all_cards = cursor.fetchall()
            
            # Calculate match scores for each card
            for card in all_cards:
                score = 0
                max_score = 0
                
                # Score name matches
                if name:
                    max_score += 100
                    card_name = card['name'].lower() if card['name'] else ""
                    search_name = name.lower()
                    
                    if FUZZY_AVAILABLE:
                        # Use fuzzy matching
                        name_score = fuzz.ratio(search_name, card_name)
                        score += name_score
                    else:
                        # Simple contains matching
                        if search_name in card_name or card_name in search_name:
                            score += 70  # Partial match
                            if search_name == card_name:
                                score += 30  # Exact match
                
                # Score attack matches
                if attacks:
                    max_score += 100
                    try:
                        card_attacks = json.loads(card['attacks']) if card['attacks'] else []
                        if card_attacks:
                            attack_scores = []
                            
                            for search_attack in attacks:
                                search_attack = search_attack.lower()
                                best_attack_score = 0
                                
                                for card_attack in card_attacks:
                                    attack_name = card_attack.get('name', '').lower()
                                    
                                    if FUZZY_AVAILABLE:
                                        attack_score = fuzz.ratio(search_attack, attack_name)
                                        best_attack_score = max(best_attack_score, attack_score)
                                    else:
                                        if search_attack in attack_name or attack_name in search_attack:
                                            best_attack_score = max(best_attack_score, 70)
                                            if search_attack == attack_name:
                                                best_attack_score = 100
                                
                                attack_scores.append(best_attack_score)
                            
                            # Average the attack scores
                            if attack_scores:
                                avg_attack_score = sum(attack_scores) / len(attack_scores)
                                score += avg_attack_score
                    except (json.JSONDecodeError, TypeError):
                        # Invalid attacks JSON or None
                        pass
                
                # Calculate percentage relevance
                relevance = score / max_score * 100 if max_score > 0 else 0
                
                # Add to matches if above threshold
                if relevance >= threshold:
                    matches.append({
                        'card': dict(card),
                        'relevance': relevance
                    })
            
            # Sort by relevance (highest first)
            matches.sort(key=lambda x: x['relevance'], reverse=True)
            
            # Filter by number if provided AND we have multiple matches
            if number and len(matches) > 1:
                number_matches = [m for m in matches if str(m['card'].get('number', '')).strip() == str(number).strip()]
                if number_matches:
                    matches = number_matches
            
            # Return matches
            return matches
            
        except sqlite3.Error as e:
            self.ocr_text.insert(tk.END, f"\nDatabase error: {e}\n")
            return []
        finally:
            if conn:
                conn.close()
    
    def on_db_result_select(self, event):
        """Handler for when a card is selected in the DB results tree"""
        selected_items = self.results_tree.selection()
        if not selected_items:
            return

        item_id = selected_items[0]
        # The 'text' of the tree item is the card ID we stored
        selected_card_id_str = self.results_tree.item(item_id, "text") 
        
        # Find the corresponding card data in our stored DB search results
        selected_card_data = None
        if hasattr(self, 'search_results') and self.search_results:
            for match in self.search_results:
                # Compare string representation of DB ID with tree item text
                if str(match['card'].get('id', '')) == selected_card_id_str:
                    selected_card_data = match['card']
                    relevance = match['relevance']
                    break
        
        if selected_card_data:
            print(f"DB Tree Selection: {selected_card_data.get('name', '?')} (Relevance: {relevance:.1f}%)")
            self.display_card_match(selected_card_data, self.db_match_canvas, 'db_match_photo', f"DB Match (Rel: {relevance:.1f}%)")
        else:
            print(f"Could not find data for selected DB result ID: {selected_card_id_str}")
            self.db_match_canvas.delete("all")
            self.db_match_canvas.create_text(10, 10, text="Error finding selected card data.", anchor=tk.NW, fill="red")

    def display_card_match(self, card_info, canvas, photo_attr_name, title_prefix="Match"):
        """Displays a matched card's details and image on a specific canvas."""
        canvas.delete("all")

        # Find the image path robustly
        image_path = self.find_card_image_path(card_info)

        # Display card text details (use determined set_name)
        set_name = card_info.get('set_name') or card_info.get('set')
        number = card_info.get('number')
        card_text = (
            f"{title_prefix}\n"
            f"Name: {card_info.get('name', 'Unknown')}\n"
            f"Set: {set_name or 'Unknown'}\n"
            f"Number: {number or 'Unknown'}"
        )
        canvas.create_text(
            10, 10, text=card_text, anchor=tk.NW, 
            font=("Arial", 10), fill="black" # Smaller font for matches
        )
            
        if not image_path:
            error_msg = "Card image not found."
            canvas.create_text(
                10, 60, text=error_msg, anchor=tk.NW, # Position below text
                font=("Arial", 10), fill="red"
            )
            return
            
        try:
            # Load image using PIL
            img = Image.open(image_path).convert("RGB")
            
            # Display image on the specified canvas (reuse logic)
            # Calculate available height after text
            canvas.update_idletasks()
            text_height_approx = 60 # Estimate height needed for text
            available_height = canvas.winfo_height() - text_height_approx
            available_width = canvas.winfo_width() - 20 # Padding

            img_w, img_h = img.size
            if img_w == 0 or img_h == 0 or available_width <=0 or available_height <=0:
                 raise ValueError("Invalid image or canvas dimensions for scaling")

            scale_w = available_width / img_w
            scale_h = available_height / img_h
            scale = min(scale_w, scale_h, 1.0) 

            new_width = int(img_w * scale)
            new_height = int(img_h * scale)
            
            if new_width <= 0: new_width = 1
            if new_height <= 0: new_height = 1
                
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and store
            photo_image = ImageTk.PhotoImage(resized_img)
            setattr(self, photo_attr_name, photo_image) # Store e.g., self.clip_match_photo
            
            # Display the image below the text
            canvas.create_image(
                10, text_height_approx, # Position below text
                image=photo_image, anchor=tk.NW
            )
            
        except Exception as e:
            error_text = f"Error displaying card image:\n{str(e)}"
            print(error_text)
            # Display error message below text details
            canvas.create_text(
                10, 60, text=error_text, anchor=tk.NW,
                font=("Arial", 10), fill="red", width=canvas.winfo_width()-20
            )

    def find_card_image_path(self, card_info):
        """Robustly find the filesystem path for a card image, prioritizing translated DB paths."""
        image_path = None
        
        # --- Start: Prioritize DB Path Translation --- 
        db_path_to_try = card_info.get('image_large_local') or card_info.get('image_small_local')
        
        if db_path_to_try:
            # Normalize path separators for consistent splitting
            db_path_to_try = db_path_to_try.replace('\\', '/')
            
            # Attempt to extract set and filename (expecting format like '.../set_name/filename.png')
            parts = db_path_to_try.strip('/').split('/')
            if len(parts) >= 2:
                db_filename = parts[-1]
                db_set_name = parts[-2]
                
                # Construct the potential filesystem path
                potential_path = os.path.join(FILESYSTEM_IMAGE_PATH, db_set_name, db_filename)
                
                if os.path.exists(potential_path):
                    print(f"Found image translating DB path: {potential_path}")
                    return potential_path
                else:
                    print(f"Translated DB path not found: {potential_path} (from DB path: {db_path_to_try})")
            else:
                 print(f"Could not parse set/filename from DB path: {db_path_to_try}")
        # --- End: Prioritize DB Path Translation ---

        # --- Fallback Methods --- 
        # Get details needed for fallbacks
        set_name_from_source = card_info.get('set_name') or card_info.get('set') 
        number = card_info.get('number')
        filename_from_meta = card_info.get('filename')
        
        if not set_name_from_source:
             print(f"Warning: Missing set name for fallbacks (Card: {card_info.get('name')}-{number}).")
             # Cannot proceed with fallbacks if set name is missing
             return None 

        # Fallback 1: Try using filename from metadata (CLIP) + set name from source
        if filename_from_meta:
             base_filename = os.path.basename(filename_from_meta) 
             potential_path = os.path.join(FILESYSTEM_IMAGE_PATH, set_name_from_source, base_filename)
             if os.path.exists(potential_path):
                  print(f"Found image using metadata filename fallback: {potential_path}")
                  return potential_path

        # Fallback 2: Try constructing from set name (from source) and number
        if number:
            variations = [f"{number}_large.png", f"{number}.png", f"{number}.jpg", 
                          f"{number}_small.png"]
            for var in variations:
                potential_path = os.path.join(FILESYSTEM_IMAGE_PATH, set_name_from_source, var)
                if os.path.exists(potential_path):
                    print(f"Found image constructing path from source set/number fallback: {potential_path}")
                    return potential_path
                    
        # If none of the above worked
        print(f"Image not found after all checks for card: Set='{set_name_from_source}', Number='{number}', DBPathTried='{db_path_to_try}'")
        print(f"  (Searched primary location: {os.path.join(FILESYSTEM_IMAGE_PATH, set_name_from_source if set_name_from_source else '')})")
        return None

    def update_combined_match(self):
        """Determine the best combined match using simplified rules."""
        self.combined_match_canvas.delete("all")

        clip_match_via_faiss = self.best_clip_match_info # Best visual match from FAISS index
        db_matches = self.search_results # Text-based matches from DB
        input_embedding = self.input_image_embedding # Embedding of the dropped image

        chosen_card_info = None
        title = "Combined Match"
        
        # Get all 100% relevance matches
        perfect_matches = [match for match in db_matches if match['relevance'] == 100.0]
        
        if len(perfect_matches) == 1:
            # Rule 1: Single 100% match - use it
            chosen_card_info = perfect_matches[0]['card']
            title = f"Combined Match (Perfect DB Match)"
            print(f"Using single 100% DB match -> {chosen_card_info.get('name')}")
            
        elif len(perfect_matches) > 1 and clip_match_via_faiss and input_embedding is not None:
            # Rule 2: Multiple 100% matches - use CLIP to choose between them
            print(f"Found {len(perfect_matches)} perfect DB matches, using CLIP to choose")
            
            best_similarity = -1
            best_match = None
            
            for match in perfect_matches:
                card_info = match['card']
                card_path = self.find_card_image_path(card_info)
                if card_path:
                    card_embedding = self.get_clip_embedding(card_path)
                    if card_embedding is not None:
                        similarity = 1.0 - np.linalg.norm(input_embedding - card_embedding) / 1.5
                        similarity = max(0.0, min(1.0, similarity)) * 100.0
                        print(f"  - Perfect match {card_info.get('name')}: CLIP similarity {similarity:.1f}%")
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = card_info
            
            if best_match:
                chosen_card_info = best_match
                title = f"Combined Match (Perfect DB + CLIP {best_similarity:.1f}%)"
                print(f"Chose perfect match with highest CLIP similarity -> {chosen_card_info.get('name')}")
            else:
                # Fallback to CLIP if comparison fails
                chosen_card_info = clip_match_via_faiss
                title = f"Combined Match (CLIP - Perfect compare failed)"
                print(f"Perfect match comparison failed, using CLIP -> {chosen_card_info.get('name')}")
            
        elif clip_match_via_faiss:
            # Rule 3: No 100% matches - use CLIP
            chosen_card_info = clip_match_via_faiss
            clip_similarity = None
            
            # Calculate CLIP similarity for display if possible
            if input_embedding is not None:
                clip_path = self.find_card_image_path(clip_match_via_faiss)
                if clip_path:
                    clip_embedding = self.get_clip_embedding(clip_path)
                    if clip_embedding is not None:
                        similarity = 1.0 - np.linalg.norm(input_embedding - clip_embedding) / 1.5
                        clip_similarity = max(0.0, min(1.0, similarity)) * 100.0
            
            # Show best DB relevance in title if available
            best_db_relevance = max([match['relevance'] for match in db_matches]) if db_matches else None
            
            if clip_similarity is not None and best_db_relevance is not None:
                title = f"Combined Match (CLIP {clip_similarity:.1f}% > DB {best_db_relevance:.1f}%)"
            elif clip_similarity is not None:
                title = f"Combined Match (CLIP {clip_similarity:.1f}%)"
            else:
                title = "Combined Match (CLIP Only)"
                
            print(f"No perfect DB matches, using CLIP -> {chosen_card_info.get('name')}")
        
        else:
            print("No matches found (neither perfect DB nor CLIP)")
            self.combined_match_canvas.create_text(10, 10, text="No match found.", anchor=tk.NW, fill="red")
            return

        # Display the chosen card
        if chosen_card_info:
            self.display_card_match(chosen_card_info, self.combined_match_canvas, 'combined_match_photo', title)
        else:
            self.combined_match_canvas.create_text(10, 10, text="No match found.", anchor=tk.NW, fill="red")

if __name__ == "__main__":
    # Use TkinterDnD Tk object
    root = TkinterDnD.Tk() 
    app = YoloImageDetector(root)
    root.mainloop() 