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
    - [Prerequisites](#prerequisites)
    - [Clone the Repository](#clone-the-repository)
    - [Create and Activate a Virtual Environment](#create-and-activate-a-virtual-environment)
    - [Install Dependencies](#install-dependencies)
  - [Usage](#usage)
    - [1. Data Processing, Dataset Inspection, and Model Training](#1-data-processing-dataset-inspection-and-model-training)
    - [2. Extract Image Embeddings](#2-extract-image-embeddings)
    - [3. Process a Single Image](#3-process-a-single-image)
    - [4. Similarity Search Using FAISS](#4-similarity-search-using-faiss)
  - [Additional Notes](#additional-notes)

---

## **📌 Project Overview**
This project implements an end-to-end pipeline for processing raw product and image data, training a transfer learning model based on a modified ResNet50, extracting image embeddings, and performing similarity search using FAISS. The goal is to enhance the recommendation and ranking system for Facebook Marketplace by leveraging both image and text data.

It aims to improve **product ranking & recommendations** by leveraging **both image & text data** from product listings.

---

## File Structure

- **`a_main.py`**  
  Processes raw data, inspects the dataset, and trains the transfer learning model using the `ResNetTransferLearner`.

- **`a_resnet_transfer_trainer.py`**  
  Implements the full transfer learning training pipeline (data processing, DataLoader setup, model configuration, training loop, and saving model weights/metrics).

- **`b_extract_embeddings.py`**  
  Extracts image embeddings for every valid image in the `cleaned_images` folder and saves them as a JSON file (`image_embeddings.json`).

  - **`b_feature_extractor_model.py`**  
  Contains the shared feature extraction model (a modified ResNet50) and the transformation pipeline.

- **`b_image_processor.py`**  
  Processes the first valid image found in the `cleaned_images` folder by applying the training transformations and adding a batch dimension. This prepares the image to be fed to the model and prints the processed tensor's shape.

- **`c_faiss_search.py`**  
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

✅ **Step 5: Text Embeddings (CNN)**
- Convert text into vector form using:
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
- Deploy on **AWS and FastAPI** to serve search queries.
- Optimize for **low-latency inference**.

✅ **Step 11: Live Recommendations**
- Store **user interaction history**.
- Apply **collaborative filtering + content-based filtering** for **personalized ranking**.

---

## **🚀 Technologies & Tools**
| **Component**       | **Technology Used** |
|---------------------|--------------------|
| **📊 Data Processing**  | `pandas`, `numpy`, `scikit-learn` |
| **🖼️ Image Processing**  | `PIL`, `torchvision` |
| **🧠 Deep Learning**  | `PyTorch` (ResNet, CNN) |
| **📡 Search Indexing**  | `FAISS` (Vector Search) |
| **🖥️ Deployment**  | `Docker`, `FastAPI`, `AWS` |

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

### Prerequisites

- **Python 3.13+**
- A virtual environment manager (e.g., Conda or venv)
- Required Python packages (see `requirements.txt`)

### Clone the Repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### Create and Activate a Virtual Environment

Using Conda:
```bash
conda create --name marketplace_env python=3.13.1
conda activate marketplace_env
```

Or using venv:
```bash
python -m venv marketplace_env
source marketplace_env/bin/activate   # On Unix/macOS
marketplace_env\Scripts\activate      # On Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Data Processing, Dataset Inspection, and Model Training

Run the main script to process raw data, inspect the dataset, and train the transfer learning model:

```bash
python main.py
```

**What It Does:**

- Uses `ProductLabeler` to process raw product and image data.
- Generates a training CSV file at `data/training_data.csv`.
- Inspects the dataset (prints encoder/decoder mappings, sample batches, total sample count, and an example sample).
- Trains the transfer learning model using `ResNetTransferLearner` (defined in `a_resnet_transfer_trainer.py`).
- Saves model weights and training metrics.

### 2. Extract Image Embeddings

After training, extract embeddings for every image in the `cleaned_images` folder:

```bash
python extract_embeddings.py
```

**What It Does:**

- Uses the shared feature extraction model to process every valid image in `cleaned_images`.
- Creates a dictionary mapping each image id (filename without extension) to its 1000-dimensional embedding.
- Saves the dictionary as `image_embeddings.json`.

### 3. Process a Single Image

To process only the first valid image from the `cleaned_images` folder (for model input):

```bash
python image_processor.py
```

**What It Does:**

- Scans the `cleaned_images` folder for valid images.
- Automatically selects the first image (alphabetically).
- Applies the training transformation pipeline and adds a batch dimension.
- Prints the shape of the processed image tensor (should be `(1, 3, 256, 256)`).

### 4. Similarity Search Using FAISS

To perform a similarity search with a query image:

```bash
python faiss_search.py
```

**What It Does:**

- Loads image embeddings from `image_embeddings.json`.
- Builds a FAISS index from the embeddings.
- Automatically selects the first valid image from `cleaned_images` as the query.
- Extracts the query image embedding and uses FAISS to retrieve the top 5 similar images.
- Prints the image IDs and distances of the similar images.

---

## Additional Notes

- **Model Weights:**  
  The project expects saved model weights in `final_model/image_model.pt`. If these weights are not found, default model parameters will be used.

- **Folder Structure:**  
  - Place raw data files (e.g., `Cleaned_Products.csv` and `Images.csv`) in the `data/` folder.
  - Store all product images in the `cleaned_images/` folder.
  - Keep model weights in the `final_model/` folder.

- **Customization:**  
  Adjust hyperparameters (batch size, number of epochs, learning rate, etc.) in the `ResNetTransferLearner` class in `a_resnet_transfer_trainer.py` as needed.

- **Common Module:**  
  The feature extraction model and transformation pipeline are centralized in `feature_extractor_model.py` to avoid duplication.

---