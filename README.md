## 🚗 Vehicle Insurance Claim Prediction — End-to-End MLOps Project

> **A full-scale MLOps implementation** that takes raw data from MongoDB, builds machine learning pipelines for prediction, and deploys the model automatically using AWS, Docker, and GitHub Actions CI/CD.

---

### 🧭 Table of Contents

* [🚀 Overview](#-overview)
* [🧩 Tech Stack](#-tech-stack)
* [⚙️ Setup & Environment](#️-setup--environment)
* [💾 MongoDB Atlas Integration](#-mongodb-atlas-integration)
* [🏗️ MLOps System Design & Workflow](#️-mlops-system-design--workflow)
* [🤖 Model Training & Evaluation](#-model-training--evaluation)
* [☁️ AWS & S3 Model Registry](#️-aws--s3-model-registry)
* [🐳 CI/CD Pipeline (GitHub Actions + Docker + EC2)](#-cicd-pipeline-github-actions--docker--ec2)
* [🎯 Core Features](#-core-features)
* [📈 Results & Future Scope](#-results--future-scope)
* [👨‍💻 Author](#-author)

---

## 🚀 Overview

This project automates the **entire lifecycle of a Machine Learning system** — from **data collection to model deployment** — for a **Vehicle Insurance Claim Prediction** use case.

It is built following MLOps principles to ensure:

* Continuous integration and delivery (CI/CD)
* Scalable and reproducible model training
* Cloud storage and automated deployment
* Monitoring, validation, and model versioning

The system is designed for production-readiness — replicating the architecture used in enterprise ML pipelines.

---

## 🧩 Tech Stack

| Category                   | Tools & Technologies                |
| -------------------------- | ----------------------------------- |
| **Programming Language**   | Python 3.10                         |
| **Machine Learning**       | Scikit-learn, NumPy, Pandas         |
| **Database**               | MongoDB Atlas                       |
| **Cloud Platform**         | AWS (S3, EC2, ECR, IAM)             |
| **Containerization**       | Docker                              |
| **Automation**             | GitHub Actions (CI/CD)              |
| **Web Framework**          | Flask                               |
| **Logging & Monitoring**   | Custom Logging + Exception Handling |
| **Environment Management** | Conda                               |

---

## ⚙️ Setup & Environment

### 1️⃣ Create Project Template

```bash
python template.py
```

### 2️⃣ Package Management

* Add local package imports in `setup.py` and `pyproject.toml`
* Reference: `crashcourse.txt`

### 3️⃣ Virtual Environment

```bash
conda create -n vehicle python=3.10 -y
conda activate vehicle
pip install -r requirements.txt
pip list  # confirm local packages installed
```

---

## 💾 MongoDB Atlas Integration

1. Create a free cluster on [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
2. Set up a **Database User** (username & password).
3. Allow network access:

   * Add IP: `0.0.0.0/0` (open for all)
4. Copy **Connection String**:

   ```
   mongodb+srv://<username>:<password>@cluster.mongodb.net/
   ```
5. Save it as an environment variable:

   ```bash
   export MONGODB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net/"
   ```
6. Upload dataset via Jupyter notebook → `mongoDB_demo.ipynb`
7. Verify data in Atlas Dashboard under *Browse Collections*

---

## 🏗️ MLOps System Design & Workflow

The architecture is modular, scalable, and production-grade — each layer is independent yet seamlessly integrated.

### 🧱 1. **Data Layer**

* **Source:** Raw structured data stored in MongoDB Atlas.
* **Access:** Data is fetched using a connector built in `mongo_db_connection.py`.
* **Transformation:** Data converted from key–value JSON format to clean Pandas DataFrame.
* **Artifacts:** Saved in an “artifact” folder (version-controlled and ignored via `.gitignore`).

---

### ⚙️ 2. **Pipeline Layer**

This layer orchestrates all major steps of the ML lifecycle:

1. **Data Ingestion**

   * Pulls data from MongoDB
   * Converts JSON → CSV
   * Stores train-test split locally

2. **Data Validation**

   * Validates column names, datatypes, and schema (`config/schema.yaml`)
   * Logs anomalies and missing fields

3. **Data Transformation**

   * Encodes categorical variables
   * Scales numerical features
   * Saves preprocessed arrays for model input

4. **Model Training**

   * Trains models (e.g., Random Forest, Decision Tree)
   * Selects best model based on accuracy/F1-score
   * Saves trained model locally and logs metrics

5. **Model Evaluation**

   * Compares current and previous model performance
   * Applies thresholding to decide model promotion

6. **Model Pusher**

   * Uploads the promoted model to AWS S3 bucket
   * Maintains a model registry for version tracking

---

### 🧠 3. **Entity Layer**

Defines structured Python classes for:

* Configurations (`config_entity.py`)
* Artifacts between components (`artifact_entity.py`)
* Model and S3 estimators (`estimator.py`, `s3_estimator.py`)

These abstractions ensure **type safety**, **traceability**, and **code reusability**.

---

### 🧾 4. **Utility & Logging Layer**

* `main_utils.py` – shared helper functions for file I/O, schema validation, etc.
* `logger.py` – unified logging across all modules
* `exception.py` – custom exception tracking with root-cause trace

---

### 🌐 5. **Application Layer**

* A lightweight **Flask web app (`app.py`)** serves model predictions.
* Routes:

  * `/train` → triggers model training pipeline
  * `/predict` → accepts user input & returns prediction

---

## 🤖 Model Training & Evaluation

**Workflow Summary:**

1. Train model on ingested + transformed data
2. Validate model accuracy & drift against previous versions
3. Auto-promote if accuracy > threshold (default: 0.02 improvement)
4. Push model to **S3 registry** if passed evaluation

---

## ☁️ AWS & S3 Model Registry

1. Create IAM User → attach `AdministratorAccess`
2. Store credentials:

   ```bash
   export AWS_ACCESS_KEY_ID="XXXX"
   export AWS_SECRET_ACCESS_KEY="XXXX"
   ```
3. Create S3 Bucket: `my-model-mlopsproj`
4. Constants:

   ```python
   MODEL_BUCKET_NAME = "my-model-mlopsproj"
   MODEL_PUSHER_S3_KEY = "model-registry"
   MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02
   ```
5. Code managed under:

   * `src/configuration/aws_connection.py`
   * `src/aws_storage/`
   * `src/entity/s3_estimator.py`

---

## 🐳 CI/CD Pipeline (GitHub Actions + Docker + EC2)

### 🧰 Steps:

1. **Dockerize** the entire project

   * `Dockerfile` and `.dockerignore` configured
   * Container exposes port `5080`

2. **GitHub Actions Workflow**

   * File: `.github/workflows/aws.yaml`
   * Builds Docker image → Pushes to AWS ECR → Deploys on EC2

3. **AWS Setup**

   * **ECR:** Store container image
   * **EC2:** Host the Flask app
   * **IAM User:** For automation permissions
   * **Security Group:** Allow inbound rule on port 5080

4. **GitHub Secrets:**

   ```
   AWS_ACCESS_KEY_ID
   AWS_SECRET_ACCESS_KEY
   AWS_DEFAULT_REGION
   ECR_REPO
   ```

5. **Trigger CI/CD:**

   * Every push to main branch triggers build → deploy → serve
   * Access live app via:

     ```
     http://<EC2_Public_IP>:5080
     ```

---

## 🎯 Core Features

✅ Fully Modular & Extensible MLOps Design
✅ Automated MongoDB → Pandas → ML Pipeline
✅ Centralized Logging & Exception Management
✅ Dynamic Data Validation via Schema
✅ Cloud Storage (AWS S3) Model Registry
✅ Dockerized Deployment for Reproducibility
✅ Continuous Integration/Deployment with GitHub Actions
✅ Flask Web Interface for Real-time Predictions

---

## 📈 Results & Future Scope

**Achievements:**

* Built a fully automated ML workflow
* Integrated CI/CD, cloud, and containerization
* Deployed model endpoint on AWS EC2

**Next Steps:**

* Add **MLflow** for model tracking
* Integrate **Prometheus & Grafana** for monitoring
* Enable **Auto Retraining** on data drift detection

---

## 👨‍💻 Author

**Karan Singh**
📧 [[your.email@example.com](mailto:your.email@example.com)]
💼 [LinkedIn](https://www.linkedin.com/in/your-profile)
📂 [GitHub](https://github.com/your-github)

---

> ⭐ *If this project inspired you, consider giving it a star on GitHub!*

---


