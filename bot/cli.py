import argparse
import os
import requests
import csv
import re

from PIL import Image

parser = argparse.ArgumentParser(
                    prog='InnoBot CLI',
                    description='Performs CLI requests to backend')

parser.add_argument('path')
args = parser.parse_args()
print(args.path)
pathes = []
full_pathes = []
for file in os.listdir(args.path):
    if file[0] == '.' or file.split(".")[-1] not in ["jpg", "jpeg"]:
        continue
    pathes.append(file)
    full_path = os.path.join(args.path, file)
    full_pathes.append(full_path)

api_url = "http://localhost:5000/process_images"


# image_files = [("files", open("test_images/3d_mask.jpg", "rb")),
#                ("files", open("test_images/real.jpg", "rb"))]

image_files = [("files", open(x, "rb")) for x in full_pathes]

response = requests.post(api_url, files=image_files)
r_json = response.json()
print(r_json)
with open("out.csv", "w", newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    for i, label in enumerate(r_json["msg"]):
        spamwriter.writerow([pathes[i], label])

