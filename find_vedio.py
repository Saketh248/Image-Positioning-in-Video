# we can easily position of img in the vedio...


#####generating frames and stores in frames(list) 

import cv2

def extract_frames(video_path, interval=1):#functio to creating frames 
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames

# Extract frames from the specified video path
video_path = "/media/video1.mp4"
frames = extract_frames(video_path)





####    extracting feautures of images to identify easily 





from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np

# Load the VGG16 model pre-trained on ImageNet
model = VGG16(weights='imagenet', include_top=False)

def extract_features(img):
    img = cv2.resize(img, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    return features.flatten()

# Load the query image
query_img_path = "/media/Screenshot (50).png"
query_img = cv2.imread(query_img_path)
query_features = extract_features(query_img)





 
 ##### By using cosine similarity  it is easy to find the best and same img from vedio..






from sklearn.metrics.pairwise import cosine_similarity
from google.colab.patches import cv2_imshow
def find_best_match(query_features, frames):
    best_match = None
    best_similarity = -1
    for frame in frames:
        frame_features = extract_features(frame)
        similarity = cosine_similarity([query_features], [frame_features])[0][0] ##cosine similarity
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = frame
    return best_match

# Find the best matching frame
best_match = find_best_match(query_features, frames)

# Display the best matching frame (if needed)
if best_match is not None:
    cv2_imshow(best_match)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No matching frame found.")
