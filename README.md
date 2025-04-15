# Pokemon Card Detector

A client-server application that detects and identifies Pokemon cards. The application uses a combination of computer vision, OCR, and image similarity to identify cards.

## Features

- Detects Pokemon cards in images using YOLO object detection
- Extracts text from card regions using PaddleOCR
- Matches cards using both visual similarity (CLIP) and text matching
- Provides a combined match based on the best available evidence
- Client-server architecture for distributed processing

## Server Component

The server processes card images and returns detection and matching results. It runs on CPU by default.

### Server Requirements

- Python 3.8 or higher
- PyTorch and other dependencies listed in `requirements.txt`
- YOLO models: `models/detector/best.pt` and `models/classifier/best.pt`
- CLIP resources: `data/clip/card_vectors_clip_vitb32.faiss` and `data/clip/card_ids_clip_vitb32.json`
- Card database: `data/pokemon_cards_simple.db`
- Card images at `data/assets/cards/`

### Running the Server

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the server:
   ```bash
   python server.py
   ```

   The server will listen on http://localhost:8000 by default.

## Client Component

The client provides a user interface for uploading images and viewing results.

### Client Requirements

- Python 3.8 or higher
- Tkinter and tkinterdnd2
- Other dependencies listed in `requirements.txt`

### Running the Client

1. Make sure the server is running first.

2. Start the client:
   ```bash
   python client.py
   ```

3. Drag and drop a Pokemon card image onto the input canvas to process it.

### Client Features

- Drag-and-drop interface for easy image uploading
- Displays the input image, CLIP match, database match, and combined match
- Shows detailed OCR results and detection information
- Displays a list of all database matches with relevance scores
- Ability to select any match from the list to view details

## API Documentation

The server exposes the following API endpoints:

- `POST /api/detect_card`: Upload an image to detect and identify a Pokemon card

Visit http://localhost:8000/docs for interactive API documentation.

## Customizing

- **Server URL**: If running the server on a different machine or port, edit the `SERVER_URL` variable in `client.py`.
- **YOLO Models**: Replace the models in the `models` directory to use your own trained models.
- **Database**: The application uses a SQLite database. You can replace it with your own or extend the schema.

## Troubleshooting

- If you encounter issues with the YOLO models loading, check that the PyTorch version is compatible.
- For OCR issues, make sure PaddleOCR and PaddlePaddle are properly installed.
- If the client cannot connect to the server, verify the server is running and the URL is correct.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 