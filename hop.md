Below is a complete **USAGE.md** file provided as one markdown code snippet:

```markdown
# Facebook Marketplace Recommendation & Ranking System

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
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
- [Conclusion](#conclusion)

---

## Overview

This project implements an end-to-end pipeline for processing raw product and image data, training a transfer learning model based on a modified ResNet50, extracting image embeddings, and performing similarity search using FAISS. The goal is to enhance the recommendation and ranking system for Facebook Marketplace by leveraging both image and text data.

---

## Project Structure

- **`feature_extractor_model.py`**  
  Contains the shared feature extraction model (a modified ResNet50) and the transformation pipeline.

- **`main.py`**  
  Processes raw data, inspects the dataset, and trains the transfer learning model using the `ResNetTransferLearner`.

- **`a_resnet_transfer_trainer.py`**  
  Implements the full transfer learning training pipeline (data processing, DataLoader setup, model configuration, training loop, and saving model weights/metrics).

- **`extract_embeddings.py`**  
  Extracts image embeddings for every valid image in the `cleaned_images` folder and saves them as a JSON file (`image_embeddings.json`).

- **`image_processor.py`**  
  Processes the first valid image found in the `cleaned_images` folder by applying the training transformations and adding a batch dimension. This script is used to prepare a single image as model input.

- **`faiss_search.py`**  
  Loads saved image embeddings, builds a FAISS index, and performs a similarity search using a query image (automatically selected from `cleaned_images`).

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

## Conclusion

This project provides a complete pipeline—from data processing and model training to image embedding extraction and similarity search using FAISS. Follow the installation and usage instructions above to run the entire system and customize it for your needs. If you have any questions or encounter issues, please refer to the documentation in the source code or contact the project maintainers.

Happy coding!
```

---

Save the above snippet as **USAGE.md** in your repository. This file contains all the details regarding installation, usage, and project structure in one complete markdown code snippet.