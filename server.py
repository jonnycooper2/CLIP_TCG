import os
# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from ultralytics import YOLO
import sqlite3
import json
import re
import faiss
import torch
import open_clip
from PIL import Image
import io
import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import tempfile
import shutil
from pathlib import Path
import gc
import concurrent.futures

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
DB_IMAGE_PATH = "assets/images"  # What's in the database
FILESYSTEM_IMAGE_PATH = "data/assets/images"  # Where files are actually stored - was cards, now images

# CLIP model paths
FAISS_INDEX_PATH = "data/clip/card_vectors_clip_vitb32.faiss"
CARD_IDS_PATH = "data/clip/card_ids_clip_vitb32.json"

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

# Define response models
class CardMatch(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    set_name: Optional[str] = None
    number: Optional[str] = None
    attacks: Optional[List[str]] = None
    relevance: Optional[float] = None
    image_base64: Optional[str] = None
    # Add fields to store the original DB image paths
    local_large_image: Optional[str] = None
    local_small_image: Optional[str] = None
    
class DetectionResults(BaseModel):
    detected_regions: List[str] = []
    ocr_name: Optional[str] = None
    ocr_attacks: List[str] = []
    ocr_number: Optional[str] = None
    
class CardResponse(BaseModel):
    success: bool
    message: str
    detection_results: Optional[DetectionResults] = None
    clip_match: Optional[CardMatch] = None
    db_match: Optional[CardMatch] = None
    combined_match: Optional[CardMatch] = None
    db_matches: List[CardMatch] = []

# Create the FastAPI app
app = FastAPI(title="Pokemon Card Detector API")

# Add CORS middleware for the client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models and resources
detector_model = None
classifier_model = None
clip_model = None
clip_preprocess = None
clip_index = None
clip_card_ids = None
ocr = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and resources on server startup"""
    global detector_model, classifier_model, clip_model, clip_preprocess, clip_index, clip_card_ids, ocr
    
    # Force CPU for all models
    device = "cpu"
    torch.device(device)
    
    # Load YOLO models with compatibility handling for PyTorch 2.6+
    detector_model_path = r"models/detector/best.pt"
    classifier_model_path = r"models/classifier/best.pt"
    
    try:
        # First try with safe globals approach
        detector_model = YOLO(detector_model_path)
        classifier_model = YOLO(classifier_model_path)
        print("Successfully loaded YOLO models with safe globals")
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
            detector_model = YOLO(detector_model_path)
            classifier_model = YOLO(classifier_model_path)
            print("Successfully loaded models with weights_only=False")
        finally:
            # Restore original function
            torch.load = original_load
    
    # Initialize PaddleOCR if available
    if PADDLE_AVAILABLE:
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)
    
    # Load CLIP resources
    try:
        print("Loading CLIP (ViT-B/32) model...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='laion2b_s34b_b79k'
        )
        clip_model.to(device)
        clip_model.eval()
        print(f"CLIP model loaded successfully onto {device}.")

        print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
        if not os.path.exists(FAISS_INDEX_PATH):
            print(f"Warning: FAISS index file not found: {FAISS_INDEX_PATH}")
        else:
            clip_index = faiss.read_index(FAISS_INDEX_PATH)
            print(f"FAISS index loaded ({clip_index.ntotal} vectors).")

        print(f"Loading card metadata from {CARD_IDS_PATH}...")
        if not os.path.exists(CARD_IDS_PATH):
            print(f"Warning: Card ID file not found: {CARD_IDS_PATH}")
        else:
            with open(CARD_IDS_PATH, "r") as f:
                clip_card_ids = json.load(f)
                # Convert keys back to integers if they were saved as strings
                clip_card_ids = {int(k): v for k, v in clip_card_ids.items()}
            print("Card metadata loaded.")
    except Exception as e:
        print(f"Error loading CLIP resources: {e}")

def preprocess_image_for_clip(image_path):
    """Load and preprocess image using CLIP's preprocessor"""
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Use CLIP's preprocessor
        img_tensor = clip_preprocess(img).unsqueeze(0)  # Add batch dim
        return img_tensor
    except Exception as e:
        print(f"Error preprocessing image for CLIP: {e}")
        return None
        
def get_clip_embedding(image_path):
    """Generate a normalized CLIP embedding for a given image path."""
    if not clip_model or not clip_preprocess:
        print("CLIP model/preprocessor not loaded.")
        return None

    # Preprocess image
    img_tensor = preprocess_image_for_clip(image_path)
    if img_tensor is None:
        return None

    # Get embedding
    try:
        with torch.no_grad():
            embedding_tensor = clip_model.encode_image(img_tensor)
            # Normalize
            embedding_tensor /= embedding_tensor.norm(dim=-1, keepdim=True)
            # Move to CPU and convert to NumPy array
            embedding = embedding_tensor.cpu().numpy()[0]
        return embedding
    except Exception as e:
        print(f"Failed to generate CLIP embedding: {e}")
        return None

def find_closest_card_with_clip(image_path, k=3):
    """Find the top k closest matching cards using CLIP embeddings and FAISS."""
    if not clip_index or not clip_card_ids:
        print("CLIP index/metadata not loaded. Skipping FAISS search.")
        return []

    # Get embedding for the query image
    query_embedding = get_clip_embedding(image_path)
    if query_embedding is None:
        print(f"Could not get CLIP embedding for query image")
        return []

    # Prepare for FAISS search
    embedding_matrix = np.array([query_embedding]).astype("float32")

    # Distances are L2 distances, lower is better
    distances, indices = clip_index.search(embedding_matrix, k)

    matches = []
    for i in range(k):
        if i >= len(indices[0]) or indices[0][i] == -1:
            print(f"Match {i+1}: Invalid index found.")
            break
        nearest_index = indices[0][i]
        distance = distances[0][i]

        if nearest_index in clip_card_ids:
            match_info = clip_card_ids[nearest_index]
            matches.append((match_info, distance, query_embedding))
        else:
            print(f"Warning: Index {nearest_index} not found in metadata")
    return matches

def extract_text_from_regions(img, result):
    """Extract text using OCR from regions detected by YOLO."""
    if not PADDLE_AVAILABLE or ocr is None:
        return None, [], None
        
    height, width = img.shape[:2]
    boxes = result.boxes
    class_names = result.names
    
    if len(boxes) == 0:
        return None, [], None
    
    detected_name = None
    detected_attacks = []
    detected_number = None
    detected_regions = []
    
    # Process each detected box
    for i, box in enumerate(boxes):
        class_id = int(box.cls[0].item())
        class_name = class_names[class_id]
        confidence = float(box.conf[0].item())
        
        detected_regions.append(f"{class_name} ({confidence:.2f})")
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width-1, x2), min(height-1, y2)
        
        # Extract region
        region = img[y1:y2, x1:x2]
        
        # Skip if region is empty
        if region.size == 0:
            continue
        
        # Run OCR on the region
        try:
            ocr_result = ocr.ocr(region, cls=True)
            
            extracted_text = ""
            
            # Parse OCR results
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
                                concatenated_text += text + " "
                                confidences.append(confidence)
                    
                    # Use the concatenated text if any
                    if concatenated_text:
                        extracted_text = concatenated_text.strip()
            
            # Store the detected text based on class
            if extracted_text:
                if class_name == "name":
                    detected_name = extracted_text
                elif class_name == "attack":
                    detected_attacks.append(extracted_text)
                elif class_name == "number":
                    # Try to extract just the first number from a format like "xxx/yyy"
                    number_match = re.search(r'(\d+)\s*\/?\s*\d*', extracted_text)
                    if number_match:
                        detected_number = number_match.group(1)
                    else:
                        # Fallback to the raw text if no pattern match
                        detected_number = extracted_text.strip()
                
        except Exception as e:
            print(f"{class_name} region: Error performing OCR: {str(e)}")
    
    return detected_name, detected_attacks, detected_number, detected_regions

def search_pokemon_by_name_and_attacks(name=None, attacks=None, number=None, threshold=60):
    """
    Search for Pokemon cards by name, attacks, and optionally number
    using fuzzy matching to handle OCR imperfections.
    """
    if not name and not attacks:
        return []
    
    # Connect to database
    if not os.path.exists(DB_PATH):
        print(f"Error: Cannot find database at '{DB_PATH}'")
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
            relevance_np = score / max_score * 100 if max_score > 0 else 0
            # Cast to standard Python float before appending
            relevance = float(relevance_np) 
            
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
        print(f"Database error: {e}")
        return []
    finally:
        if conn:
            conn.close()

def find_card_image_path(card_info: Union[Dict, CardMatch]):
    """Robustly find the filesystem path for a card image, prioritizing translated DB paths."""
    # Check the type of card_info and access data accordingly
    if isinstance(card_info, CardMatch):
        # Access attributes using dot notation for CardMatch objects
        card_name = card_info.name or 'Unknown'
        card_number = card_info.number or 'Unknown'
        card_set = card_info.set_name or 'Unknown' # Assuming set_name is populated
        db_path_to_try = card_info.local_large_image or card_info.local_small_image
        filename_from_meta = None # CardMatch doesn't store this directly from CLIP metadata
        set_name_from_source = card_set
    elif isinstance(card_info, dict):
        # Access attributes using .get() for dictionaries
        card_name = card_info.get('name', 'Unknown')
        card_number = card_info.get('number', 'Unknown')
        card_set = card_info.get('set_name') or card_info.get('set', 'Unknown')
        db_path_to_try = card_info.get('local_large_image') or card_info.get('local_small_image') or \
                         card_info.get('image_large_local') or card_info.get('image_small_local') # Check original names too
        filename_from_meta = card_info.get('filename')
        set_name_from_source = card_set
    else:
        print(f"Error: Unexpected type for card_info: {type(card_info)}")
        return None

    print(f"Looking for image for: {card_name} #{card_number} from {card_set}")
    
    # --- Start: Prioritize DB Path Translation --- 
    print(f"  DB path: {db_path_to_try}")
    
    if db_path_to_try:
        # Fix path structure differences:
        # Convert "assets/images/hgss3/33_large.png" to "data/assets/images/hgss3/33_large.png"
        if db_path_to_try.startswith(DB_IMAGE_PATH + "/"):
            # Extract the part after "assets/images/"
            suffix = db_path_to_try[len(DB_IMAGE_PATH)+1:]
            # Construct path with the correct structure
            full_path = os.path.join(FILESYSTEM_IMAGE_PATH, suffix)
            print(f"  Checking path: {full_path}")
            if os.path.exists(full_path):
                print(f"  Found image at: {full_path}")
                return full_path
            else:
                print(f"  Path not found: {full_path}")

    # --- Fallback Methods --- 
    # Removed redundant gets, variables are set above based on type
    
    if not set_name_from_source:
        return None 

    # Fallback 1: Try set codes directly if set name is in format with dash/hyphen
    if set_name_from_source and "-" in set_name_from_source:
        # Try to extract the set code from set names like "HS—Undaunted"
        set_code = None
        # Common pattern: 2-letter code followed by hyphen or em-dash
        match = re.search(r'^([A-Za-z]{2})[\-—]', set_name_from_source)
        if match:
            set_code = match.group(1).lower()
            if card_number:
                # Try a few naming patterns for set codes
                potential_codes = [
                    f"{set_code}ss{card_number}", # For HGSS sets
                    f"{set_code}{card_number}",   # Generic pattern
                ]
                
                for code in potential_codes:
                    for ext in ["_large.png", ".png", ".jpg", "_small.png"]:
                        path = os.path.join(FILESYSTEM_IMAGE_PATH, code, card_number + ext)
                        print(f"  Checking special path: {path}")
                        if os.path.exists(path):
                            print(f"  Found image at: {path}")
                            return path

    # Fallback 2: Try using filename from metadata (CLIP) + set name from source
    if filename_from_meta: # This only applies if card_info was a dict
        base_filename = os.path.basename(filename_from_meta) 
        # Try direct set name
        potential_path = os.path.join(FILESYSTEM_IMAGE_PATH, set_name_from_source, base_filename)
        print(f"  Checking metadata path: {potential_path}")
        if os.path.exists(potential_path):
            print(f"  Found image at: {potential_path}")
            return potential_path
            
        # Try lowercase version
        if set_name_from_source:
            set_code = set_name_from_source.lower().replace(" ", "").replace("—", "-")
            potential_path = os.path.join(FILESYSTEM_IMAGE_PATH, set_code, base_filename)
            print(f"  Checking metadata path (lowercase): {potential_path}")
            if os.path.exists(potential_path):
                print(f"  Found image at: {potential_path}")
                return potential_path

    # Fallback 3: For HGSS series specifically (since we're seeing issues with Raichu from HS—Undaunted)
    if "HS—Undaunted" in str(set_name_from_source) and card_number:
        potential_path = os.path.join(FILESYSTEM_IMAGE_PATH, "hgss3", f"{card_number}_large.png")
        print(f"  Checking HGSS special path: {potential_path}")
        if os.path.exists(potential_path):
            print(f"  Found image at: {potential_path}")
            return potential_path

    # Fallback 4: Try constructing from set name (from source) and number
    if card_number:
        variations = [f"{card_number}_large.png", f"{card_number}.png", f"{card_number}.jpg", 
                     f"{card_number}_small.png"]
        for var in variations:
            potential_path = os.path.join(FILESYSTEM_IMAGE_PATH, set_name_from_source, var)
            print(f"  Checking path: {potential_path}")
            if os.path.exists(potential_path):
                print(f"  Found image at: {potential_path}")
                return potential_path
                
    # Final fallback: Try various set codes from the folder list
    if card_number:
        possible_folders = [
            "hgss1", "hgss2", "hgss3", "hgss4", # HeartGold & SoulSilver sets
            "base1", "base2", "base3",           # Base sets
            "sm1", "sm2", "sm3", "sm4",          # Sun & Moon sets
            "swsh1", "swsh2", "swsh3",           # Sword & Shield sets
            "xy1", "xy2", "xy3",                 # XY sets
            # Add other potential mappings based on your card database
        ]
        
        for folder in possible_folders:
            for var in [f"{card_number}_large.png", f"{card_number}.png", f"{card_number}.jpg", f"{card_number}_small.png"]:
                potential_path = os.path.join(FILESYSTEM_IMAGE_PATH, folder, var)
                print(f"  Checking fallback path: {potential_path}")
                if os.path.exists(potential_path):
                    print(f"  Found image at: {potential_path}")
                    return potential_path
    
    print(f"  No image found for {card_name} #{card_number} from {card_set}")
    return None

def get_card_image_base64(card_info):
    """Find the card image and convert it to base64 for sending to client"""
    image_path = find_card_image_path(card_info)
    if not image_path or not os.path.exists(image_path):
        return None
        
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            return base64.b64encode(img_data).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def format_card_match(card_info, relevance=None, input_embedding=None):
    """Format a card match for the API response"""
    if not card_info:
        return None
        
    # Get attack names from the attacks JSON
    attack_names = []
    try:
        attacks = card_info.get('attacks', '[]')
        # Check if attacks is already a list or dict
        if isinstance(attacks, (list, dict)):
            attacks_json = attacks
        else:
            # It's a string, so parse it
            attacks_json = json.loads(attacks)
            
        if attacks_json:
            attack_names = [a.get('name', 'Unknown') for a in attacks_json]
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # Get the card image as base64
    image_base64 = get_card_image_base64(card_info)
    
    # Format the response
    return CardMatch(
        id=str(card_info.get('id', '')),
        name=card_info.get('name', 'Unknown'),
        set_name=card_info.get('set_name', card_info.get('set', 'Unknown')),
        number=card_info.get('number', 'Unknown'),
        attacks=attack_names,
        relevance=float(relevance) if relevance is not None else None,
        image_base64=image_base64,
        # Store the original DB paths in the model
        local_large_image=card_info.get('local_large_image'),
        local_small_image=card_info.get('local_small_image')
    )

def determine_combined_match(clip_match_info, db_matches: List[CardMatch], input_embedding):
    """Determine the best combined match using rules similar to the original app"""
    if not db_matches and not clip_match_info:
        return None, "No matches found"
    
    # Get all 100% relevance matches using dot notation
    perfect_matches = [match for match in db_matches if match.relevance == 100.0]
    
    if len(perfect_matches) == 1:
        # Rule 1: Single 100% match - use it
        # perfect_matches[0] is a CardMatch object, pass it directly
        chosen_card_info = perfect_matches[0]
        return chosen_card_info, "Perfect DB Match"
        
    elif len(perfect_matches) > 1 and clip_match_info and input_embedding is not None:
        # Rule 2: Multiple 100% matches - use CLIP to choose between them
        best_similarity = -1
        best_match_object = None
        
        for match_obj in perfect_matches:
            # match_obj is a CardMatch object, use its attributes
            card_path = find_card_image_path(match_obj) # Pass the object
            if card_path:
                card_embedding = get_clip_embedding(card_path)
                if card_embedding is not None:
                    similarity_np = 1.0 - np.linalg.norm(input_embedding - card_embedding) / 1.5
                    # Cast similarity to standard float
                    similarity = float(max(0.0, min(1.0, similarity_np)) * 100.0)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_object = match_obj # Store the whole object
        
        if best_match_object:
            # Return the chosen CardMatch object
            return best_match_object, f"Perfect DB + CLIP {best_similarity:.1f}%"
        else:
            # Fallback to CLIP if comparison fails
            # clip_match_info is dict, format it
            return format_card_match(clip_match_info, None), "CLIP (Perfect compare failed)"
        
    elif clip_match_info:
        # Rule 3: No 100% matches - use CLIP
        # clip_match_info is dict from CLIP search result, format it
        clip_formatted = format_card_match(clip_match_info, None) 
        clip_similarity = None
        
        # Calculate CLIP similarity for display if possible
        if input_embedding is not None and clip_formatted:
            # Use the formatted object to find path
            clip_path = find_card_image_path(clip_formatted)
            if clip_path:
                clip_embedding = get_clip_embedding(clip_path)
                if clip_embedding is not None:
                    similarity_np = 1.0 - np.linalg.norm(input_embedding - clip_embedding) / 1.5
                    # Cast similarity to standard float
                    clip_similarity = float(max(0.0, min(1.0, similarity_np)) * 100.0)
                    # Update relevance in the formatted object if calculated
                    clip_formatted.relevance = clip_similarity
        
        # Show best DB relevance in title if available (use dot notation)
        best_db_relevance = max([match.relevance for match in db_matches if match.relevance is not None]) if db_matches else None
        
        if clip_similarity is not None and best_db_relevance is not None:
            reason = f"CLIP {clip_similarity:.1f}% > DB {best_db_relevance:.1f}%"
        elif clip_similarity is not None:
            reason = f"CLIP {clip_similarity:.1f}%"
        else:
            reason = "CLIP Only"
            
        return clip_formatted, reason # Return the formatted CardMatch object
    
    else:
        # Case where only db_matches exist but none are perfect
        if db_matches:
            # Return the top DB match (which is a CardMatch object)
            return db_matches[0], f"Best DB Match ({db_matches[0].relevance:.1f}%) "
        else:
            return None, "No matches found"

def run_clip_pipeline(cropped_path):
    """Runs the CLIP similarity check pipeline."""
    try:
        clip_matches = find_closest_card_with_clip(cropped_path, k=1)
        clip_match_info = None
        clip_match = None
        input_embedding = None
        
        if clip_matches:
            clip_match_info, distance, input_embedding = clip_matches[0]
            similarity = float(max(0, min(100, (1 - distance/2.0) * 100))) # Cast to float
            clip_match = format_card_match(clip_match_info, similarity) # Pass info dict
        
        return clip_match_info, clip_match, input_embedding
    except Exception as e:
        print(f"Error in CLIP pipeline: {e}")
        # Return None for all expected values on error
        return None, None, None

def run_ocr_db_pipeline(crop_image_cv, classifier_model):
    """Runs the OCR extraction and Database search pipeline."""
    try:
        # Step 3: Run classifier model for OCR regions
        classifier_results = classifier_model(crop_image_cv)
        
        if not classifier_results or len(classifier_results) == 0:
            print("Classification failed or returned empty results.")
            detection_results = DetectionResults(detected_regions=["Classification failed"])
            return detection_results, None, []
            
        # Step 4: Extract text from regions
        detected_name, detected_attacks, detected_number, detected_regions = extract_text_from_regions(
            crop_image_cv, classifier_results[0]
        )
        
        # Create detection results object first
        detection_results = DetectionResults(
            detected_regions=detected_regions or ["No regions detected"],
            ocr_name=detected_name,
            ocr_attacks=detected_attacks,
            ocr_number=detected_number
        )
        
        # Step 5: Search database using OCR results
        db_matches_formatted = []
        db_match_top = None
        
        if detected_name or detected_attacks or detected_number:
            search_results = search_pokemon_by_name_and_attacks(
                name=detected_name,
                attacks=detected_attacks,
                number=detected_number,
                threshold=60
            )
            
            # Format DB matches for response
            for match in search_results:
                card = match['card']
                relevance = float(match['relevance']) # Ensure float
                # Pass the original card dict to format_card_match
                formatted = format_card_match(card, relevance)
                if formatted:
                     db_matches_formatted.append(formatted)
            
            # Set top DB match if available
            if db_matches_formatted:
                db_match_top = db_matches_formatted[0]
        
        return detection_results, db_match_top, db_matches_formatted
    except Exception as e:
        print(f"Error in OCR/DB pipeline: {e}")
        # Return default values on error
        return DetectionResults(detected_regions=["OCR/DB Error"]), None, []

@app.post("/api/detect_card", response_model=CardResponse)
async def detect_card(file: UploadFile = File(...)):
    """Process an uploaded card image and return detection results"""
    if not detector_model or not classifier_model:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Server models not initialized"}
        )
    
    # Create a temporary file to save the uploaded image
    # Use try-with-resources for tempfile
    temp_path = None
    cropped_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)
            
        # Process the image
        img_pil = Image.open(temp_path).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Step 1: Detect card using detector model
        detector_results = detector_model(img_cv, conf=0.25)
        
        if not detector_results or len(detector_results[0].boxes) == 0:
            return CardResponse(
                success=True,
                message="No cards detected in the image",
                detection_results=DetectionResults(detected_regions=["No cards detected"])
            )
        
        # Process the first detected card
        box = detector_results[0].boxes[0]
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
        
        # Save the cropped image for further processing
        # Ensure unique name in case multiple requests happen concurrently
        cropped_path = f"{temp_path}_crop_{os.getpid()}.jpg"
        cv2.imwrite(cropped_path, crop)

        # Parallel Execution using ThreadPoolExecutor
        clip_match_info = None
        clip_match = None
        input_embedding = None
        detection_results = None
        db_match = None
        db_matches = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit tasks
            clip_future = executor.submit(run_clip_pipeline, cropped_path)
            ocr_db_future = executor.submit(run_ocr_db_pipeline, crop, classifier_model)

            # Retrieve results
            try:
                clip_match_info, clip_match, input_embedding = clip_future.result()
            except Exception as e:
                print(f"CLIP pipeline future failed: {e}")
                # Handle potential error from the future itself
                clip_match_info, clip_match, input_embedding = None, None, None

            try:
                detection_results, db_match, db_matches = ocr_db_future.result()
            except Exception as e:
                print(f"OCR/DB pipeline future failed: {e}")
                # Handle potential error from the future itself
                detection_results = DetectionResults(detected_regions=["OCR/DB Error"])
                db_match, db_matches = None, []

        # Ensure detection_results has a default value if OCR/DB failed badly
        if detection_results is None:
             detection_results = DetectionResults(detected_regions=["Processing Error"])

        # Step 6: Determine combined match
        # Pass the full db_matches list (which now contains original paths via CardMatch objects)
        combined_match, combined_reason = determine_combined_match(
            clip_match_info, 
            db_matches, # Pass the list of CardMatch models directly
            input_embedding
        )
        
        # Prepare response
        response = CardResponse(
            success=True,
            message=f"Card processed successfully. Combined match: {combined_reason}",
            detection_results=detection_results,
            clip_match=clip_match,
            db_match=db_match,
            combined_match=combined_match,
            db_matches=db_matches
        )
        
        # Collect garbage and empty cache
        gc.collect()
        # torch.cuda.empty_cache() # Not needed as we forced CPU
        
        return response
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"ERROR PROCESSING CARD: {str(e)}\n{error_detail}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error processing image: {str(e)}"}
        )
    finally:
        # Clean up temporary files robustly
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as unlink_e:
                print(f"Error removing temp file {temp_path}: {unlink_e}")
        if cropped_path and os.path.exists(cropped_path):
            try:
                os.unlink(cropped_path)
            except Exception as unlink_e:
                 print(f"Error removing cropped file {cropped_path}: {unlink_e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 