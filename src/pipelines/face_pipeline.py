import numpy as np
import dlib
import face_recognition_models
import streamlit as st

from src.database.db import get_all_students

@st.cache_resource
def load_dlib_models():
    """
    Loads and caches dlib face detector, shape predictor, and face recognition models.
    Cached so models are only loaded once into memory.
    """
    detector = dlib.get_frontal_face_detector()

    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec

def get_face_embeddings(image_np):
    """
    Detects all faces in an image and computes a 128-dimensional embedding vector for each face.
    """
    # Safeguard: If the image has an alpha channel (RGBA), drop it to keep RGB (3 channels)
    if image_np.ndim == 3 and image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    detector, sp, facerec = load_dlib_models()
    faces = detector(image_np, 1)

    encodings = []

    for face in faces:
        shape = sp(image_np, face)
        face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1)
        encodings.append(np.array(face_descriptor, dtype=np.float64))

    return encodings

@st.cache_resource
def get_train_model():
    """
    Fetches all registered students and their face embeddings from the database.
    Cached using st.cache_resource to avoid fetching from Supabase on every frame.
    """
    embeddings = []
    student_ids = []

    student_db = get_all_students()

    if not student_db:
        return None

    for student in student_db:
        embedding = student.get('face_embedding')
        if embedding:
            embeddings.append(np.array(embedding, dtype=np.float64))
            student_ids.append(student.get('student_id'))

    if len(embeddings) == 0:
        return None

    return {
        'embeddings': np.array(embeddings),  # Shape: (N, 128)
        'student_ids': student_ids
    }

def train_classifier():
    """
    Clears the cached face embeddings so newly registered students are loaded into memory.
    """
    st.cache_resource.clear()
    model_data = get_train_model()
    return bool(model_data)

def predict_attendance(class_image_np, resemblance_threshold=0.48):
    """
    Scans an image for faces and matches them against registered students.

    How matching works:
    1. Extracts 128-dimensional face embeddings for all faces detected in class_image_np.
    2. Calculates the Euclidean distance between each detected face and ALL registered students.
    3. Finds the student with the smallest distance (best match).
    4. Only accepts the match if best_distance <= resemblance_threshold (0.48).
       - Lower threshold (e.g. 0.48) prevents false positives (recognizing someone else).
       - If distance > 0.48, the face is rejected as unrecognized.

    Returns:
    - detected_students: dict of {student_id: True} for verified students
    - all_students: list of all registered student IDs in the database
    - num_faces: total count of faces detected in the image
    """
    encodings = get_face_embeddings(class_image_np)
    detected_students = {}

    model_data = get_train_model()

    if not model_data or len(encodings) == 0:
        all_ids = model_data['student_ids'] if model_data else []
        return detected_students, all_ids, len(encodings)

    known_embeddings = model_data['embeddings']  # Matrix of shape (N, 128)
    student_ids = model_data['student_ids']      # List of student IDs corresponding to each row
    all_students = sorted(list(set(student_ids)))

    for encoding in encodings:
        # Calculate Euclidean distance between this face and all known embeddings
        distances = np.linalg.norm(known_embeddings - encoding, axis=1)

        # Find the student who has the smallest distance to this face
        best_match_idx = int(np.argmin(distances))
        best_match_score = float(distances[best_match_idx])

        # Strict threshold check: Only accept if the distance is below the threshold
        if best_match_score <= resemblance_threshold:
            matched_id = student_ids[best_match_idx]
            detected_students[matched_id] = True

    return detected_students, all_students, len(encodings)


