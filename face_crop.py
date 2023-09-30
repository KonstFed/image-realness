import numpy as np
import face_recognition

def crop_face(image: np.ndarray) -> np.ndarray:
    faces = face_recognition.face_locations(image)

    if not len(faces):
        return image

    face = faces[0]

    top, right, bottom, left = face
    faceImage = image[top:bottom, left:right]
    # final = Image.fromarray(faceImage)

    return faceImage