# Facebook Marketplace Recommendation & Ranking System

## Table of Contents
- [Facebook Marketplace Recommendation \& Ranking System](#facebook-marketplace-recommendation--ranking-system)
  - [Table of Contents](#table-of-contents)
  - [**📌 Project Overview**](#-project-overview)
  - [File Structure](#file-structure)
  - [**📌 Project Roadmap**](#-project-roadmap)
    - [\*\* 1. Data Collection \& Preprocessing\*\*](#-1-data-collection--preprocessing)
    - [\*\* 2. Feature Extraction\*\*](#-2-feature-extraction)
    - [\*\* 3. Search Index \& Ranking System\*\*](#-3-search-index--ranking-system)
    - [\*\* 4. Real-Time Deployment\*\*](#-4-real-time-deployment)
  - [**🚀 Technologies \& Tools**](#-technologies--tools)
  - [**📝 Next Steps**](#-next-steps)
  - [Installation](#installation)
    - [**Prerequisites**](#prerequisites)
    - [**Step 1: Clone the Repository**](#step-1-clone-the-repository)
    - [**Step 2: Create a Virtual Environment**](#step-2-create-a-virtual-environment)
    - [**Step 3: Install dependancies**](#step-3-install-dependancies)

---

## **📌 Project Overview**
This project implements an end-to-end pipeline for processing raw product and image data, training a transfer learning model based on a modified ResNet50, extracting image embeddings, and performing similarity search using FAISS. The goal is to enhance the recommendation and ranking system for Facebook Marketplace by leveraging both image and text data.

It aims to improve **product ranking & recommendations** by leveraging **both image & text data** from product listings.

---

## File Structure

- **`feature_extractor_model.py`**  
  Contains the shared feature extraction model (a modified ResNet50) and the transformation pipeline.

- **`main.py`**  
  Processes raw data, inspects the dataset, and trains the transfer learning model using the `ResNetTransferLearner`.

- **`a_resnet_transfer_trainer.py`**  
  Implements the full transfer learning training pipeline (data processing, DataLoader setup, model configuration, training loop, and saving model weights/metrics).

- **`extract_embeddings.py`**  
  Extracts image embeddings for every valid image in the `cleaned_images` folder and saves them as a JSON file (`image_embeddings.json`).

- **`image_processor.py`**  
  Processes the first valid image found in the `cleaned_images` folder by applying the training transformations and adding a batch dimension. This prepares the image to be fed to the model and prints the processed tensor's shape.

- **`faiss_search.py`**  
  Loads saved image embeddings, builds a FAISS index, and performs a similarity search using a query image (automatically selected from `cleaned_images`).

---

## **📌 Project Roadmap**
This project involves multiple stages, including **data preprocessing, feature extraction, model training, and ranking optimization**. Below is a structured implementation plan:

### ** 1. Data Collection & Preprocessing**
✅ **Step 1: Tabular Data Cleaning**
- Standardize **product listings** (e.g., prices, categories, locations).
- Extract **text features** from product descriptions.
- Encode product categories for supervised learning.

✅ **Step 2: Image Preprocessing**
- Resize images to **256x256** pixels.
- Convert all images to **RGB** to ensure uniform input.
- Apply **data augmentation** (if needed) to improve generalization.

✅ **Step 3: Merging Data**
- Link images with their corresponding product descriptions.
- Create a **single dataset** that contains both **images** and **text features**.

---

### ** 2. Feature Extraction**
✅ **Step 4: Image Embeddings (CNN)**
- Use a **pre-trained ResNet-50** (or EfficientNet) to extract meaningful **image embeddings**.
- Apply **transfer learning** by fine-tuning ResNet-50 on Marketplace categories.

✅ **Step 5: Text Embeddings (CNN or Transformer)**
- Convert text into vector form using:
  - **Word2Vec / FastText** for static embeddings.
  - **BERT / DistilBERT** for contextual embeddings.
  - **CNN** for local feature extraction from text.

✅ **Step 6: Multi-Modal Fusion**
- Concatenate **image embeddings** and **text embeddings**.
- Train a **shallow feedforward network** to classify products based on combined embeddings.

---

### ** 3. Search Index & Ranking System**
✅ **Step 7: Create the Search Index**
- Store **product embeddings** in a **vector database** (e.g., **FAISS**).
- Use **FAISS Approximate Nearest Neighbors (ANN)** for fast retrieval.

✅ **Step 8: Query Embeddings**
- Convert **user search queries** into an embedding using the trained **text model**.
- Retrieve top **K** nearest product embeddings from the search index.

✅ **Step 9: Final Ranking**
- Apply a **re-ranking model** (Gradient Boosted Trees, LightGBM, or Neural Networks).
- Incorporate **user preferences & personalization** into ranking.

---

### ** 4. Real-Time Deployment**
✅ **Step 10: Deploy as an API**
- Containerize the **ranking model** using **Docker**.
- Deploy on **AWS Lambda, GCP, or FastAPI** to serve search queries.
- Optimize for **low-latency inference**.

✅ **Step 11: Live Recommendations**
- Store **user interaction history**.
- Apply **collaborative filtering + content-based filtering** for **personalized ranking**.

---

## **🚀 Technologies & Tools**
| **Component**       | **Technology Used** |
|---------------------|--------------------|
| **📊 Data Processing**  | `pandas`, `numpy`, `scikit-learn` |
| **🖼️ Image Processing**  | `OpenCV`, `PIL`, `torchvision` |
| **🧠 Deep Learning**  | `PyTorch` (ResNet, CNN, Transformers) |
| **📡 Search Indexing**  | `FAISS` (Vector Search), `Annoy` |
| **🖥️ Deployment**  | `Docker`, `FastAPI`, `AWS/GCP` |

---

## **📝 Next Steps**
1. **Start by implementing data cleaning & preprocessing** (`main.py`).
2. **Train models for feature extraction (CNN + Transformer)**.
3. **Integrate FAISS for efficient search ranking**.
4. **Deploy the ranking system as a real-time API**.

---

This project is a **real-world application** of **multi-modal AI + search indexing**.  
Let me know if you need help with specific components! 🚀🔥

---

## Installation

### **Prerequisites**
Ensure you have **Python 3.13+** and **Conda** or **pip** installed.

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/jose-zothner-meyer/Facebook-marketplace-recommendation-ranking-system.git
cd Facebook-marketplace-recommendation-ranking-system
```

### **Step 2: Create a Virtual Environment**
```bash
conda create --name marketplace_env python=3.13.1
conda activate marketplace_env
```

### **Step 3: Install dependancies**
```bash
pip install -r requirements.txt
```