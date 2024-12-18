import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18
from torch.nn.functional import cosine_similarity

# Configuration
training_dir = "training_data"
threshold = 0.7  # Similarity threshold for recognition


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


model = resnet18(pretrained=True)
model.fc = torch.nn.Identity()  
model.eval()


def load_training_data():
    embeddings = []
    labels = []
    for person in os.listdir(training_dir):
        person_dir = os.path.join(training_dir, person)
        if os.path.isdir(person_dir):
            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    embedding = extract_embedding(img)
                    if embedding is not None:
                        embeddings.append(embedding)
                        labels.append(person)
    return embeddings, labels


def extract_embedding(image):
    try:
        tensor = transform(image).unsqueeze(0)  
        with torch.no_grad():
            embedding = model(tensor).squeeze(0)  
        return embedding
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None


known_embeddings, known_labels = load_training_data()


def run_face_recognition():
    global known_embeddings, known_labels
    cap = cv2.VideoCapture(0)

    frame_count = 0  
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 15 == 0:  
           
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_embedding = extract_embedding(face)

                if face_embedding is not None:
                    
                    if known_embeddings: 
                        similarities = [cosine_similarity(face_embedding.unsqueeze(0), known_emb.unsqueeze(0)).item()
                                        for known_emb in known_embeddings]
                        best_match_index = int(np.argmax(similarities))
                        if similarities[best_match_index] > threshold:
                            name = known_labels[best_match_index]
                        else:
                            name = "Unknown"
                    else:
                        name = "Unknown"
                else:
                    name = "Unknown"

                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Save new training data if recognized
                if name != "Unknown":
                    person_dir = os.path.join(training_dir, name)
                    if not os.path.exists(person_dir):
                        os.makedirs(person_dir)
                    img_name = f"{person_dir}/img_{len(os.listdir(person_dir)) + 1}.jpg"
                    cv2.imwrite(img_name, face)

        frame_count += 1

      
        cv2.imshow("Face Recognition", frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

run_face_recognition()
