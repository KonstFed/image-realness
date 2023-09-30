import requests

# Replace with the URL of your FastAPI endpoint
api_url = "http://localhost:5000/process_images"

# Create a list of image files to upload
# files = [('files', open('images/1.png', 'rb')), ('files', open('images/2.png', 'rb'))]
image_files = [("files", open("test_images/3d_mask.jpg", "rb")),
               ("files", open("test_images/real.jpg", "rb"))]

response = requests.post(api_url, files=image_files)

# Check the response status code
if response.status_code == 200:
    # Request was successful, print the response content
    print("Response:")
    print(response.json())
else:
    print(f"Error - Status Code: {response.status_code}")
    print(response.text)
