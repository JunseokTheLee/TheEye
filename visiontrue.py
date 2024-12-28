import os
import cv2
import torch
import numpy as np
import time#I probably don't need this, but I'm too scared to remove it for now. 
import pickle
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.nn.functional import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

# Configuration
training_dir = "training_data"
EMBEDDING_CACHE = "embeddings_cache.pkl"
threshold = 0.65#fuck
capture_delay = 8
confidence_window = 5

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, device=device, keep_all=True, min_face_size=20)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# -----------------------------
# Preprocessing and Embedding
# -----------------------------
def preprocess_face(face):
    try:
        if face is None or face.size == 0:
            return None

        if len(face.shape) == 2:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        elif face.shape[2] == 4:
            face = cv2.cvtColor(face, cv2.COLOR_BGRA2RGB)
        elif face.shape[2] == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Enhance contrast for better differentiation. Probably does jack shit, maybe not
        lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        return enhanced
    except Exception as e:
        print(f"Error in preprocess_face: {e}")
        return None


def extract_embedding(image):
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        faces = mtcnn(image)

        if faces is None or (isinstance(faces, torch.Tensor) and faces.size(0) == 0):
            return None

        face_tensor = faces[0:1].to(device)#god knows what this does

        with torch.no_grad():
            embedding = facenet(face_tensor).squeeze(0)
            embedding = embedding / embedding.norm()

        return embedding.cpu()
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None


# -----------------------------
# Training Data Management
# -----------------------------
def save_embeddings_to_cache(embeddings, labels):
    with open(EMBEDDING_CACHE, "wb") as f:
        pickle.dump({'embeddings': embeddings, 'labels': labels}, f)
    print("Embeddings cached successfully.")


def load_training_data():
    if os.path.exists(EMBEDDING_CACHE):
        print("Loading cached embeddings...")
        with open(EMBEDDING_CACHE, "rb") as f:
            data = pickle.load(f)
        return data['embeddings'], data['labels']

    print("Generating embeddings from training images...")
    embeddings, labels = [], []

    for person in os.listdir(training_dir):
        person_dir = os.path.join(training_dir, person)
        if os.path.isdir(person_dir):
            embeddings.extend(process_images_in_parallel(person_dir))
            labels.extend([person] * len(os.listdir(person_dir)))

    save_embeddings_to_cache(embeddings, labels)
    return embeddings, labels


def process_images_in_parallel(person_dir):
    embeddings = []
    image_paths = [os.path.join(person_dir, img) for img in os.listdir(person_dir)]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_and_embed_image, img_path) for img_path in image_paths]
        for future in futures:
            result = future.result()
            if result is not None:
                embeddings.append(result)

    return embeddings


def process_and_embed_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is not None:
            processed_img = preprocess_face(img)
            return extract_embedding(processed_img)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
    return None


def add_new_face_to_cache(name, face):#caching seems to be the only solution for fixing the speed issue for now. I hate it, but it somewhat works
    embedding = extract_embedding(preprocess_face(face))
    if embedding is not None:
        if os.path.exists(EMBEDDING_CACHE):
            with open(EMBEDDING_CACHE, "rb") as f:
                data = pickle.load(f)
            data['embeddings'].append(embedding)
            data['labels'].append(name)
            with open(EMBEDDING_CACHE, "wb") as f:
                pickle.dump(data, f)
            print(f"New embedding added for {name}.")


# -----------------------------
# Face Recognition System
# -----------------------------
class FaceRecognitionSystem:
    def __init__(self):
        self.known_embeddings, self.known_labels = load_training_data()
        self.recent_predictions = []
        self.last_capture_time = {}
        self.unknown_count = 0

    def get_prediction(self, face_embedding):
        if not self.known_embeddings:
            return "Unknown", 0.0
        try:
            similarities = [cosine_similarity(face_embedding.unsqueeze(0), known_emb.unsqueeze(0)).item()
                            for known_emb in self.known_embeddings]
            best_match_index = int(np.argmax(similarities))
            confidence = similarities[best_match_index]
            if confidence > threshold:
                return self.known_labels[best_match_index], confidence
        except Exception as e:
            print(f"Error in get_prediction: {e}")
        return "Unknown", 0.0

    
    def run_recognition(self, training_mode=False):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open camera")
                return
    
            frame_count = 0  # Counter for frames
    
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
    
                frame_count += 1  # Increment frame counter
    
                if frame_count % 15 == 0:  # Run recognition every 15 frames to minimize resources. In the actual implementation ,we wont need to display the screen at full frame rate
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    boxes, _ = mtcnn.detect(Image.fromarray(rgb_frame))
                    frame_count = 0
                    if boxes is not None:
                        for box in boxes:
                            x, y, x2, y2 = [int(b) for b in box]
                            face = rgb_frame[y:y2, x:x2]
                            processed_face = preprocess_face(face)
                            if processed_face is not None:
                                face_embedding = extract_embedding(processed_face)
                                name, confidence = self.get_prediction(face_embedding)
                                print(f"Recognized: {name} ({confidence:.2f})")  # Print recognized person to console
    
                                if training_mode and name != "Unknown":
                                    add_new_face_to_cache(name, face)
                                    print("face added to existing person cache")
    
                                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                                cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
            

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run_recognition(training_mode=False)
