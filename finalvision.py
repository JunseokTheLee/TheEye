import os
import torch
import numpy as np
import pickle
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from picamera2 import Picamera2
from torch.nn.functional import cosine_similarity

# Configuration
training_dir = "training_data"
EMBEDDING_CACHE = "embeddings_cache.pkl"
threshold = 0.2
confidence_window = 5

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, device=device, keep_all=True, min_face_size=20)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# -----------------------------
# Preprocessing and Embedding
# -----------------------------
def preprocess_face(image):
    try:
        if image is None:
            return None

        # Convert the image to RGB if it's not already
        if len(image.shape) == 2:  # Grayscale
            image = np.stack((image,) * 3, axis=-1)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]

        return image
    except Exception as e:
        print(f"Error in preprocess_face: {e}")
        return None


def extract_embedding(image):
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        faces = mtcnn(image)

        if faces is None or (isinstance(faces, torch.Tensor) and faces.size(0) == 0):
            print("No faces detected.")
            return None

        face_tensor = faces[0:1].to(device)

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
        print(f"Embeddings: {len(data['embeddings'])}, Labels: {len(data['labels'])}")
        return data['embeddings'], data['labels']

    print("Generating embeddings from training images...")
    embeddings, labels = [], []

    for person in os.listdir(training_dir):
        person_dir = os.path.join(training_dir, person)
        if os.path.isdir(person_dir):
            print(f"Processing person: {person}")
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = Image.open(img_path)
                processed_img = preprocess_face(np.array(img))
                if processed_img is not None:
                    embedding = extract_embedding(processed_img)
                    if embedding is not None:
                        embeddings.append(embedding)
                        labels.append(person)

    save_embeddings_to_cache(embeddings, labels)
    return embeddings, labels


# -----------------------------
# Face Recognition System
# -----------------------------
class FaceRecognitionSystem:
    def __init__(self):
        self.known_embeddings, self.known_labels = load_training_data()
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
            picam2 = Picamera2()
            picam2.start()

            print("Starting recognition loop...")
            while True:
                frame = picam2.capture_array()  # Capture frame from camera
                processed_img = preprocess_face(frame)
                if processed_img is not None:
                    face_embedding = extract_embedding(processed_img)
                    if face_embedding is not None:
                        name, confidence = self.get_prediction(face_embedding)
                        print(f"Recognized: {name} ({confidence:.2f})")
                        if name == "Unknown":
                            self.unknown_count += 1
                        else:
                            self.unknown_count = 0

                        if training_mode and name != "Unknown":
                            print(f"Adding {name} to cache.")
                else:
                    print("No valid face detected.")

        except KeyboardInterrupt:
            print("Stopping recognition...")
        finally:
            picam2.stop()


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run_recognition(training_mode=False)

