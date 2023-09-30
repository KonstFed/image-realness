import numpy as np
import face_recognition
import cv2

def crop_face(image: np.ndarray) -> np.ndarray:
    faces = face_recognition.face_locations(image)

    if not len(faces):
        return image

    face = faces[0]

    top, right, bottom, left = face
    faceImage = image[top:bottom, left:right]
    # final = Image.fromarray(faceImage)

    return faceImage


def crop_resize(image, resize_to=300):
    faces = face_recognition.face_locations(image)
    
    if len(faces):
        top, right, bottom, left = faces[0]
        image = image[top:bottom, left:right]
    
    image = cv2.resize(image, [resize_to, int(resize_to * image.shape[0] / image.shape[1])])
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

if __name__ == "__main__":
    import os
    os.makedirs("datasets/test_dataset_faces/fake")
    os.makedirs("datasets/test_dataset_faces/living")
    