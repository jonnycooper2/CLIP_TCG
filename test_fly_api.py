import requests
import json

# --- Configuration ---
# IMPORTANT: Replace this with the actual URL of your deployed Fly.io app
FLY_APP_URL = "https://clip-tcg.fly.dev" # Updated with actual URL
PREDICT_ENDPOINT = f"{FLY_APP_URL}/predict"

# URL of an image to test with (replace with a valid public image URL)
TEST_IMAGE_URL = "https://i.ebayimg.com/images/g/Pe8AAOSwHgVkgVpy/s-l400.jpg" # Updated eBay Image URL
# You can find image URLs on sites like pokemontcg.io

# Number of results to request
K_VALUE = 5

def test_api():
    """Sends a test request to the deployed Fly.io API."""
    if "your-app-name.fly.dev" in FLY_APP_URL:
        print("*** Please update FLY_APP_URL in this script with your actual deployed app URL! ***")
        return

    print(f"Sending request to: {PREDICT_ENDPOINT}")
    print(f"Using image URL: {TEST_IMAGE_URL}")

    # Prepare the request payload
    payload = {
        "image_url": TEST_IMAGE_URL,
        "k": K_VALUE
    }

    try:
        # Send the POST request
        response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=30) # Added timeout

        # Check the response status code
        if response.status_code == 200:
            print("\n--- Success! ---")
            try:
                # Try to parse and print the JSON response
                results = response.json()
                print(json.dumps(results, indent=2))
            except json.JSONDecodeError:
                print("Error: Could not decode JSON response.")
                print("Raw response text:", response.text)
        else:
            print(f"\n--- Error: Received status code {response.status_code} ---")
            try:
                # Try to print error detail from JSON response
                error_data = response.json()
                print(json.dumps(error_data, indent=2))
            except json.JSONDecodeError:
                print("Could not decode error response.")
                print("Raw response text:", response.text)

    except requests.exceptions.RequestException as e:
        print(f"\n--- Request Failed --- ")
        print(f"Error connecting to the server or sending the request: {e}")
    except Exception as e:
        print(f"\n--- An unexpected error occurred --- ")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api() 