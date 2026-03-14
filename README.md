# Autonomous Abstraction for Lung Cancer Registries

This repository contains the source code and evaluation results for a research project focused on automating the abstraction of clinical information from lung cancer medical records. The goal is to leverage Large Language Models (LLMs) to improve the efficiency and accuracy of cancer registry data entry.

## 📌 Project Overview
This study evaluates and compares several state-of-the-art LLMs (Llama 3, Qwen, and Med42) for medical data extraction. It also includes a pipeline for training a "Student Model" to perform specialized clinical abstraction tasks more efficiently.

## 📂 Repository Structure

### 📓 Notebooks (Data Exploration)
- `Check_data_mimic.ipynb`: Initial analysis and validation of the MIMIC dataset.
- `data_preparation.ipynb`: Data cleaning and preprocessing pipeline.

### 🐍 Scripts (Core Pipeline)
- `create_training_dataset.py`: Generates the dataset required for model training.
- `train_student_model.py`: Script for training/fine-tuning the student model.
- `test_student_model.py`: Evaluates the performance of the trained student model.
- `v4_run_batch_processing.py`: Automated batch processing for large-scale data extraction.

### 📊 Results (CSV Data)
- `llama3_test_results.csv`: Extraction results using the Llama 3 model.
- `qwen_test_results.csv`: Extraction results using the Qwen model.
- `med42_test_results.csv`: Extraction results using the Med42 (medical-specialized) model.
- `v4_final_lung_cancer_extraction_results.csv`: Final aggregated extraction outcomes.

## 💻 Technical Setup

### Prerequisites
- **Python Version**: `3.10.12`
- **Development Environment**: Visual Studio Code (VS Code)

### Installation
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/d931111002/Autonomous-Abstraction-for-Lung-Cancer-Registries.git](https://github.com/d931111002/Autonomous-Abstraction-for-Lung-Cancer-Registries.git)

2. Navigate to the project directory:
   cd Autonomous-Abstraction-for-Lung-Cancer-Registries

3. Activate your virtual environment:
   # For Linux/macOS
   source venv_train/bin/activate

   # For Windows
   venv_train\Scripts\activate
   
4. Install the required dependencies:
   pip install -r requirements.txt
   (Note: Ensure you create a requirements.txt file or install packages like torch, transformers, and pandas manually).

## Usage
- To run the batch extraction process, use:
  python v4_run_batch_processing.py
  
- To train the student model:
  python train_student_model.py

## Data Privacy and Ethics
This repository is Private. The datasets used (including MIMIC) are handled according to ethical research guidelines.


-- Author: Dian
