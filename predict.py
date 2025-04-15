import torch
import faiss
import open_clip
import numpy as np
import json
from PIL import Image
import requests
from io import BytesIO
import os
from cog import BasePredictor, Input, Path

# --- Configuration ---
# Model paths (relative to the built Cog/Docker image root)
FAISS_INDEX_PATH = "data/clip/card_vectors_clip_vitb32.faiss"
CARD_IDS_PATH = "data/clip/card_ids_clip_vitb32.json"

# CLIP model details
CLIP_MODEL_NAME = 'ViT-B-32'
CLIP_PRETRAINED = 'laion2b_s34b_b79k'

class Predictor(BasePredictor):
    def setup(self):
        """Load the model and resources into memory to make running multiple predictions efficient"""
        print("Setting up predictor...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load CLIP model
        print(f"Loading CLIP model: {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED
        )
        self.model.to(self.device)
        self.model.eval()
        print("CLIP model loaded.")

        # Load FAISS index
        print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"FAISS index loaded ({self.index.ntotal} vectors).")

        # Load card metadata
        print(f"Loading card metadata from {CARD_IDS_PATH}...")
        if not os.path.exists(CARD_IDS_PATH):
            raise FileNotFoundError(f"Card metadata not found at {CARD_IDS_PATH}")
        with open(CARD_IDS_PATH, "r") as f:
            self.metadata = json.load(f)
            # NOTE: Assuming keys in JSON are strings. If they were saved as ints,
            # access later might need int(idx) instead of str(idx).
            # Verify based on how card_ids_clip_vitb32.json was created.
        print("Card metadata loaded.")

    def predict(
        self,
        image_url: str = Input(description="URL of the card image to process"),
        k: int = Input(description="Number of top matches to return", default=5, ge=1, le=50)
    ) -> list:
        """Runs CLIP similarity search on the input image URL and returns top K matches."""
        print(f"Received prediction request for URL: {image_url}, k={k}")

        # 1. Load image from URL
        try:
            response = requests.get(image_url)
            response.raise_for_status() # Raise an exception for bad status codes
            image = Image.open(BytesIO(response.content)).convert("RGB")
            print("Image loaded successfully from URL.")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching image from URL: {e}")
            raise ValueError(f"Could not fetch image from URL: {e}")
        except Exception as e:
            print(f"Error opening image: {e}")
            raise ValueError(f"Could not open image data: {e}")

        # 2. Preprocess and embed image
        try:
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_image(tensor)
                # Normalize
                embedding /= embedding.norm(dim=-1, keepdim=True)
                # Move to CPU for numpy conversion
                vec = embedding.cpu().numpy().astype("float32")
            print("Image embedded successfully.")
        except Exception as e:
            print(f"Error during image embedding: {e}")
            raise RuntimeError(f"Failed to embed image: {e}")

        # 3. FAISS search
        try:
            distances, indices = self.index.search(vec, k)
            print(f"FAISS search completed. Found indices: {indices[0]}")
        except Exception as e:
            print(f"Error during FAISS search: {e}")
            raise RuntimeError(f"FAISS search failed: {e}")

        # 4. Format results
        results = []
        if len(indices) > 0:
            for i, idx_int in enumerate(indices[0]):
                if idx_int == -1: # FAISS uses -1 for invalid indices
                    continue
                
                idx_str = str(idx_int) # Use string key based on common JSON practice
                
                # Check if index exists in metadata
                card_meta = self.metadata.get(idx_str)
                
                if card_meta:
                    # Calculate similarity score approximation from L2 distance
                    distance = distances[0][i]
                    # Clamp distance to avoid potential issues, max distance is sqrt(2) for normalized vectors
                    clamped_distance = max(0.0, min(distance, 2.0))
                    # Convert to similarity percentage (0-100)
                    score = float((1 - clamped_distance / 2.0) * 100)

                    # Extract best available image identifier
                    # Prioritize 'image_large_local' if present, else 'filename' or 'id'
                    image_identifier = card_meta.get('image_large_local') or card_meta.get('filename') or f"card_id_{idx_str}"

                    results.append({
                        "name": card_meta.get("name", "Unknown"),
                        "set": card_meta.get("set_name") or card_meta.get("set", "Unknown"),
                        "number": card_meta.get("number", "Unknown"),
                        "image_identifier": image_identifier, # Send back a useful path/ID
                        "score": round(score, 2),
                        "faiss_index": int(idx_int), # Include the index for debugging
                        "distance": float(distance) # Include raw distance
                    })
                else:
                     print(f"Warning: Index {idx_str} not found in metadata.")
        else:
            print("FAISS search returned no indices.")

        print(f"Returning {len(results)} results.")
        return results 