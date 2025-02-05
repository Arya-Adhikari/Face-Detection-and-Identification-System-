Face Recognition with KNN Classifier and OpenCV

This project demonstrates how to use OpenCV for face detection and recognition with a K-Nearest Neighbors (KNN) classifier, utilizing pre-trained models and custom image processing. The code captures images from the webcam, detects faces, and trains a KNN classifier to recognize faces based on embeddings generated using the MobileNetV2 model.

Requirements
Before running the script, you need to install the required libraries. You can install them using pip:
pip install opencv-python tensorflow keras scikit-learn matplotlib

Overview
Image Capture: The code captures images using a webcam. These images are saved in a directory corresponding to a person's label (e.g., "Arya").
Face Detection: The script detects faces in the captured images using the Haar Cascade face detection model from OpenCV.
Preprocessing: Detected faces are cropped and resized to a uniform size (160x160 pixels) for model compatibility.
Embedding Generation: The MobileNetV2 pre-trained model is used to generate embeddings (feature vectors) for the preprocessed face images.
KNN Classification: A KNN classifier is trained on these embeddings to identify faces.
Testing: The classifier is tested on a new image, and it predicts the identity of the face based on the trained embeddings.

How to Use
1. Capture Images
The first part of the script allows you to capture and save images from your webcam. To save an image, press s while the webcam window is open. Press q to quit the capture.

# Example usage:
label = "Arya"  # Change this for different classes
output_dir = f"./data/{label}"
2. Face Detection
Once an image is captured, the script detects faces in the image using the Haar Cascade model. Detected faces are highlighted with bounding boxes.

# Example:
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(50, 50))
3. Preprocess Faces
Detected faces are cropped and resized to 160x160 pixels. These images are saved in the directory ./processed_faces/.

# Example:
cv2.imwrite(face_path, resized_face)
4. Embedding Generation
Using the MobileNetV2 model, embeddings (feature vectors) are generated for each preprocessed face. These embeddings are used to train the KNN classifier.

# Example:
embedding = model.predict(np.expand_dims(face_preprocessed, axis=0))[0]
5. Train the KNN Classifier
The KNN classifier is trained using the embeddings and labels of the faces. The model predicts the identity of a new face based on its embedding.

# Example:
knn.fit(embeddings, labels)
6. Testing
To test the classifier on a new face, you must have a preprocessed image (test_face.jpg). The classifier will predict the identity of the face.

# Example:
predicted_label = knn.predict([new_embedding])
print("Predicted Identity:", predicted_label[0])
Folder Structure
./data/<label>: This folder contains the captured images for a specific label.
./processed_faces/: This folder contains the cropped and resized face images used for training.
./test_face.jpg: This image is used to test the trained classifier.
Troubleshooting

No Faces Detected: If no faces are detected in the captured image, ensure that the face is well-lit and clearly visible. Adjust the camera position or face detection parameters.
Model Not Found: If you're using your own pre-trained model, ensure that the model file path is correct when loading it.

Conclusion
This script demonstrates how to use OpenCV for face detection and recognition. It combines image processing with machine learning techniques, such as embeddings and KNN, to identify faces in real-time.

Feel free to modify and expand upon this project for your specific use case!
