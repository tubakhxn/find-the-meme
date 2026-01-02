
import cv2
import os
import numpy as np


# Path to memes folder
MEMES_FOLDER = os.path.join(os.path.dirname(__file__), 'memes')

# Load Haar Cascade for face detection (OpenCV built-in)
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)


# LBPH face recognizer (OpenCV built-in)
lbph = cv2.face.LBPHFaceRecognizer_create()

def load_memes():
    """Load meme images and names from the memes folder, and extract face crops for LBPH."""
    faces = []
    labels = []
    meme_images = []
    meme_names = []
    for idx, filename in enumerate(os.listdir(MEMES_FOLDER)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(MEMES_FOLDER, filename)
            img = cv2.imread(path)
            if img is None:
                continue
            face, rect = detect_face(img)
            if face is not None:
                faces.append(face)
                labels.append(idx)
                meme_images.append(img)
                meme_names.append(os.path.splitext(filename)[0])
    if len(faces) > 0:
        lbph.train(faces, np.array(labels))
    return meme_images, meme_names, faces, labels

def preprocess_face(face_img):
    """Resize and equalize face image for consistency."""
    face_img = cv2.resize(face_img, (100, 100))
    face_img = cv2.equalizeHist(face_img)
    return face_img

def detect_face(img):
    """Detect the largest face in an image and return the cropped face region and rectangle."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    # Take the largest face
    x, y, w, h = max(faces, key=lambda rect: rect[2]*rect[3])
    face_img = gray[y:y+h, x:x+w]
    face_img = preprocess_face(face_img)
    return face_img, (x, y, w, h)

def extract_webcam_face_features(frame):
    """Extract ORB features from the detected face in the webcam frame."""
    face, rect = detect_face(frame)
    if face is not None:
        kp, des = orb.detectAndCompute(face, None)
        if des is not None:
            return (kp, des), rect
    return None, None

def compare_features(des1, des2):
    """Compare two sets of ORB descriptors using BFMatcher and return the number of good matches."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    # Sort matches by distance (lower is better)
    matches = sorted(matches, key=lambda x: x.distance)
    # Count good matches (distance < 60 is a good threshold for ORB)
    good_matches = [m for m in matches if m.distance < 60]
    return len(good_matches)

def show_result(frame, meme_img, meme_name, found, face_rect):
    """Display the webcam frame and meme image side by side, with overlays."""
    display_frame = frame.copy()
    if face_rect is not None:
        x, y, w, h = face_rect
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Resize meme image to match webcam frame height
    if meme_img is not None:
        meme_img_resized = cv2.resize(meme_img, (display_frame.shape[1]//2, display_frame.shape[0]))
        combined = np.hstack((display_frame, meme_img_resized))
    else:
        # If no meme, just show webcam
        combined = display_frame
    # Add text
    if found:
        text = f"MEME FOUND 😂: {meme_name}"
        color = (0, 255, 255)
    else:
        text = "Searching for meme..."
        color = (255, 255, 255)
    cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Finding The Meme', combined)

import random

def main():
    meme_images, meme_names, faces, labels = load_memes()
    if len(meme_images) == 0:
        print("No meme faces found in the memes folder!")
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    last_meme_idx = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face, face_rect = detect_face(frame)
        if face is not None:
            # Pick a random meme when a face is detected
            meme_idx = random.randint(0, len(meme_images)-1)
            # Freeze on the same meme until face disappears
            if last_meme_idx is None:
                last_meme_idx = meme_idx
            show_result(frame, meme_images[last_meme_idx], meme_names[last_meme_idx], True, face_rect)
        else:
            last_meme_idx = None
            show_result(frame, None, '', False, None)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
