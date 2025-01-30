# Facebook Marketplace Recommendation & Ranking System

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [License](#license)

---

## Project Description

This project is designed to **process and clean data for a Facebook Marketplace Recommendation & Ranking System**.  
It focuses on cleaning both **tabular product data** and **image data** to ensure consistency before applying machine learning techniques.

### **Project Objectives**
- **Tabular Data Cleaning**: Removing missing values, standardizing prices, and encoding product categories.
- **Image Processing**: Resizing images to **512x512**, ensuring consistent color channels, and storing them in a structured format.
- **Data Integration**: Assigning category labels to images and merging them with the product dataset.
- **Feature Engineering**: Preparing the dataset for a recommendation and ranking system.

### **What You Will Learn**
- How to **clean and preprocess structured data** using Pandas.
- How to **process images** (resize, convert, and standardize formats).
- How to **merge and label datasets** for machine learning.
- How to **structure a project** for efficient data preparation.

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