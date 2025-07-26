# IMAGE_CAPTION_GENERATION_AND_CLASSIFICATION
Image Classification and Captioning System 🎯
This project integrates two core computer vision tasks—image classification and image captioning—into a unified deep learning-powered system. It allows users to upload an image and receive both a predicted class label and a descriptive caption.

🚀 Technologies Used
Frontend & Backend: Flask (Python Web Framework)
Deep Learning: TensorFlow, Keras
Image Feature Extraction: VGG16 (Pre-trained)
Image Classification: Custom CNN
Caption Generation: LSTM-based Sequence Model
Natural Language Processing: Tokenization, Sequence Modeling
Database: MySQL with SQLAlchemy
Other Tools: OpenCV, Git, Jupyter Notebook, Postman, MySQL Workbench
🧠 Features
📂 Upload images through a simple Flask web interface
🔍 Classify images into categories: Memes, Handwritten Notes, Printed Notes, Diagrams, etc.
🖼️ Generate meaningful natural language captions for each image
🧾 Store and retrieve generated captions from MySQL
🔄 Use REST API endpoints for testing and development
📊 System Architecture
📌 Modules – Captioning
Feature Extraction: VGG16 outputs a 4096-dim vector
Caption Generation: LSTM model trained on tokenized captions
Tokenizer: Converts words to sequences and vice versa
Training: LSTM model trained using extracted image features and captions
📌 Modules – Classification
Preprocessing: Resizing, normalization
Feature Extraction: Color histograms, edge detection, shape features
Classification Model: Custom CNN
Training: Image category classifier trained on labeled dataset
Image Sorting: Automatically stores images based on predicted label
🖼️ Datasets Used
Captioning: Flickr8k (or custom dataset with image-caption pairs)
Classification: Custom dataset with labeled image categories
💻 System Requirements
Hardware
CPU: Intel Core i5 or better
RAM: 8GB (16GB recommended)
GPU: NVIDIA (CUDA supported, e.g., GTX 1650+)
Storage: 50GB+ (SSD preferred)
Software
Python 3.7+
TensorFlow / Keras
Flask
OpenCV
MySQL
SQLAlchemy
📂 How to Run
Clone the Repository
git clone https://github.com/21MH1A0579/IMAGE_CAPTION_GENERATION_AND-CLASSIFICATION
cd IMAGE_CAPTION_GENERATION_AND-CLASSIFICATIONImage Classification and Captioning System 🎯
This project integrates two core computer vision tasks—image classification and image captioning—into a unified deep learning-powered system. It allows users to upload an image and receive both a predicted class label and a descriptive caption.

🚀 Technologies Used
Frontend & Backend: Flask (Python Web Framework)
Deep Learning: TensorFlow, Keras
Image Feature Extraction: VGG16 (Pre-trained)
Image Classification: Custom CNN
Caption Generation: LSTM-based Sequence Model
Natural Language Processing: Tokenization, Sequence Modeling
Database: MySQL with SQLAlchemy
Other Tools: OpenCV, Git, Jupyter Notebook, Postman, MySQL Workbench
🧠 Features
📂 Upload images through a simple Flask web interface
🔍 Classify images into categories: Memes, Handwritten Notes, Printed Notes, Diagrams, etc.
🖼️ Generate meaningful natural language captions for each image
🧾 Store and retrieve generated captions from MySQL
🔄 Use REST API endpoints for testing and development
📊 System Architecture
📌 Modules – Captioning
Feature Extraction: VGG16 outputs a 4096-dim vector
Caption Generation: LSTM model trained on tokenized captions
Tokenizer: Converts words to sequences and vice versa
Training: LSTM model trained using extracted image features and captions
📌 Modules – Classification
Preprocessing: Resizing, normalization
Feature Extraction: Color histograms, edge detection, shape features
Classification Model: Custom CNN
Training: Image category classifier trained on labeled dataset
Image Sorting: Automatically stores images based on predicted label
🖼️ Datasets Used
Captioning: Flickr8k (or custom dataset with image-caption pairs)
Classification: Custom dataset with labeled image categories
💻 System Requirements
Hardware
CPU: Intel Core i5 or better
RAM: 8GB (16GB recommended)
GPU: NVIDIA (CUDA supported, e.g., GTX 1650+)
Storage: 50GB+ (SSD preferred)
Software
Python 3.7+
TensorFlow / Keras
Flask
OpenCV
MySQL
SQLAlchemy
