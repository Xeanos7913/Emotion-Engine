import cv2
import torch
import numpy as np
from torchvision import transforms
import timm
from torch import nn
from PIL import Image

class FaceModel(nn.Module):
    
    def __init__(self):
        super(FaceModel, self).__init__()
        
        self.eff_net = timm.create_model('efficientnet_b0',
                                        pretrained = True,
                                        num_classes = 7)
        
    def forward(self, images, labels = None):
        logits = self.eff_net(images)
        
        if labels != None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return logits, loss
        
        return logits

# Path to your model
MODEL_PATH = "best-weights.pt"

# Emotion classes (adjust based on your model's training)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad','Surprise']

# Load your trained PyTorch model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FaceModel()
model.load_state_dict(torch.load('best-weights.pt', map_location=device))
model.to(device)
model.eval()

# Transformation pipeline for RGB input
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))  # Resize
        face_pil = Image.fromarray(face_resized).convert("RGB")  # Convert to RGB
        face_tensor = transform(face_pil).unsqueeze(0).to(device)  # Transform

        with torch.no_grad():
            output = model(face_tensor)
            emotion_idx = torch.argmax(output, dim=1).item()
            emotion = EMOTIONS[emotion_idx]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()