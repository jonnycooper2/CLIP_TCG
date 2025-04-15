import torch
import faiss
import open_clip
import numpy as np
import json
from PIL import Image
import requests
from io import BytesIO
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# --- Configuration ---
# Paths relative to the app directory inside the Docker container
FAISS_INDEX_PATH = "data/clip/card_vectors_clip_vitb32.faiss"
CARD_IDS_PATH = "data/clip/card_ids_clip_vitb32.json"
CLIP_MODEL_NAME = 'ViT-B-32'
CLIP_PRETRAINED = 'laion2b_s34b_b79k'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Setup --- 
class CLIPSearchModel:
    def __init__(self):
        logger.info("Initializing CLIPSearchModel...")
        # Force CPU for Fly.io deployment
        self.device = "cpu" 
        logger.info(f"Using device: {self.device}")
        
        logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        # Ensure model is on CPU
        self.model.to(self.device).eval()
        logger.info("CLIP model loaded.")

        logger.info(f"Checking for FAISS index at: {FAISS_INDEX_PATH}")
        if not os.path.exists(FAISS_INDEX_PATH):
            logger.error(f"Missing FAISS index at {FAISS_INDEX_PATH}")
            raise FileNotFoundError(f"Missing FAISS index at {FAISS_INDEX_PATH}")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        logger.info(f"FAISS index loaded ({self.index.ntotal} vectors).")

        logger.info(f"Checking for metadata at: {CARD_IDS_PATH}")
        if not os.path.exists(CARD_IDS_PATH):
            logger.error(f"Missing metadata at {CARD_IDS_PATH}")
            raise FileNotFoundError(f"Missing metadata at {CARD_IDS_PATH}")
        with open(CARD_IDS_PATH, "r") as f:
            self.metadata = json.load(f)
            # Assuming keys are strings in the JSON, consistent with previous checks
        logger.info("Card metadata loaded.")
        logger.info("CLIPSearchModel initialized successfully.")

    def predict(self, image_url: str, k: int = 5):
        logger.info(f"Predicting for URL: {image_url}, k={k}")
        # 1. Load image from URL
        try:
            resp = requests.get(image_url, timeout=10) # Added timeout
            resp.raise_for_status()
            image = Image.open(BytesIO(resp.content)).convert("RGB")
            logger.info("Image loaded from URL.")
        except Exception as e:
            logger.error(f"Failed to load image from {image_url}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")

        # 2. Embed image
        try:
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_image(tensor)
                embedding /= embedding.norm(dim=-1, keepdim=True)
                # Embedding is already on CPU, direct conversion
                vec = embedding.numpy().astype("float32") 
            logger.info("Image embedded.")
        except Exception as e:
            logger.error(f"Embedding failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

        # 3. FAISS Search
        try:
            distances, indices = self.index.search(vec, k)
            logger.info(f"FAISS search completed. Indices: {indices[0]}")
        except Exception as e:
            logger.error(f"FAISS search failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"FAISS search failed: {e}")

        # 4. Format Results
        results = []
        if len(indices) > 0:
            for i, idx_int in enumerate(indices[0]):
                if idx_int == -1:
                    continue # Skip invalid index
                    
                idx_str = str(idx_int) # Assume string keys in metadata JSON
                meta = self.metadata.get(idx_str)
                
                if not meta:
                    logger.warning(f"Metadata not found for index: {idx_str}")
                    continue # Skip if no metadata
                    
                distance = distances[0][i]
                clamped = max(0.0, min(float(distance), 2.0)) # Ensure float
                score = round((1 - clamped / 2.0) * 100, 2)
                
                # Use the correct field name 'local_large_image'
                image_identifier = meta.get("local_large_image") or meta.get("filename") or idx_str
                
                results.append({
                    "name": meta.get("name", "Unknown"),
                    "set": meta.get("set_name") or meta.get("set", "Unknown"),
                    "number": meta.get("number", "Unknown"),
                    "image_identifier": image_identifier,
                    "score": score,
                    "faiss_index": int(idx_int),
                    "distance": float(distance)
                })
        
        logger.info(f"Returning {len(results)} results.")
        return results


# --- FastAPI App --- 
app = FastAPI(title="CLIP Pokemon Card Search API")

# Load the model during startup
# NOTE: Fly.io might have multiple workers, each loading the model.
# Consider shared memory or alternative approaches for large models if memory becomes an issue.
model = CLIPSearchModel()

class Query(BaseModel):
    image_url: str
    k: int = 5

@app.post("/predict", summary="Find similar Pokemon cards using CLIP and FAISS")
def predict_endpoint(query: Query):
    """
    Accepts an image URL and returns the top K most similar cards based on CLIP embeddings.
    """
    try:
        return model.predict(query.image_url, query.k)
    except HTTPException as e:
        # Re-raise HTTPExceptions directly
        raise e
    except Exception as e:
        # Catch unexpected errors during prediction
        logger.error(f"Unexpected error during prediction for {query.image_url}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/health", summary="Health check endpoint")
def health_check():
    """Basic health check to verify the server is running."""
    # Could add checks here, e.g., model loaded status
    return {"status": "ok"}

# Optional: Add main block for local testing (though usually run via uvicorn command)
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local testing...")
    # Use port 8080 to match Dockerfile default for Fly.io
    uvicorn.run(app, host="0.0.0.0", port=8080) 