# Knowetic

### AI-Powered Deepfake Detection System

Knowetic is an AI-powered media authenticity system designed to detect manipulated and synthetic media using deep learning and computer vision.

The system focuses on detecting deepfake images and videos, explaining model predictions through visual heatmaps, and providing a real-time media analysis workflow.

> **Project Status:** 🚧 Under Development  
> **Course:** B.Sc. Artificial Intelligence & Machine Learning  
> **Project Type:** Final Year Project  
> **Team:** Individual

---

## Overview

The rapid advancement of generative AI has made it increasingly difficult to distinguish authentic media from manipulated or synthetic content.

Knowetic aims to address this problem by combining deep learning-based deepfake detection with explainable AI. The system will analyze uploaded images and videos, determine whether the media is authentic or manipulated, and provide visual explanations for its predictions.

A live media scanning module is also planned to allow users to analyze media currently being viewed on their screen.

---

## Core Modules

### 1. Deepfake Detection

A deep learning-based classifier that will be trained using the FaceForensics++ dataset to identify manipulated facial media.

**Planned capabilities:**
- Image analysis
- Video analysis
- Real/Fake classification
- Confidence score
- Frame-level analysis for videos

### 2. Explainable AI

Knowetic will use **Grad-CAM** to provide visual explanations for deep learning predictions.

The system will highlight image regions that contributed most to the model's decision, helping users understand why a particular prediction was made.

### 3. Live Media Analysis

A real-time media analysis feature designed to scan content currently being viewed by the user.

The planned workflow is:

```text
Screen / Browser Tab
        ↓
Frame Capture
        ↓
Frame Sampling
        ↓
Deepfake Detection
        ↓
Prediction + Confidence
        ↓
Live Result
```

---

## Planned System Architecture

```text
                    KNOWETIC
                       │
              ┌────────┴────────┐
              │                 │
        React Frontend     FastAPI Backend
              │                 │
              └────────┬────────┘
                       │
                  ML / DL Models
                       │
             ┌─────────┴─────────┐
             │                   │
      Deepfake Detection    Grad-CAM
      FaceForensics++       Explainability
             │
             └──────────┐
                        │
                 Live Media Scan
```

---

## Dataset

The primary dataset planned for the deepfake detection module is:

**FaceForensics++**

The dataset will be processed through a pipeline involving:

```text
Video Dataset
     ↓
Train / Validation / Test Split
     ↓
Frame Extraction
     ↓
Face Detection
     ↓
Face Alignment
     ↓
Image Resizing
     ↓
Normalization
     ↓
Model Training
```

---

## Machine Learning / Deep Learning

The project will investigate deep learning architectures suitable for deepfake detection, including:

- Xception
- EfficientNet
- ResNet

Models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

The best-performing model will be selected for the final detection pipeline.

---

## Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Backend | FastAPI |
| Frontend | React.js |
| Data Processing | NumPy, Pandas |
| Evaluation | Scikit-learn |
| Visualization | Matplotlib |
| Explainability | Grad-CAM |
| PDF Reports | ReportLab |
| Version Control | Git & GitHub |
| Deployment | Vercel, Render |
| Dataset | FaceForensics++ |

---

## Project Roadmap

### Semester 1

- [x] Development environment setup
- [x] WSL2 + NVIDIA GPU configuration
- [x] Python virtual environment
- [x] TensorFlow GPU configuration
- [x] Git repository setup
- [ ] FaceForensics++ dataset preparation
- [ ] Data preprocessing pipeline
- [ ] Deepfake detection model
- [ ] Model evaluation
- [ ] Grad-CAM explainability
- [ ] Semester 1 documentation

### Semester 2

- [ ] FastAPI backend
- [ ] React frontend
- [ ] Image upload and analysis
- [ ] Video analysis pipeline
- [ ] Live media / screen scanning
- [ ] PDF authenticity reports
- [ ] Testing and optimization
- [ ] Deployment
- [ ] Final documentation
- [ ] Final evaluation

---

## Project Goals

The final system aims to provide:

- Deepfake detection for images and videos
- Confidence-based predictions
- Explainable AI using Grad-CAM
- Frame-level video analysis
- Live media analysis
- PDF-based authenticity reports
- A user-friendly web interface

---

## Disclaimer

Deepfake detection is an evolving research problem. Model predictions should be treated as an AI-assisted assessment rather than absolute proof of authenticity.

---

## Author

**Nathaneil Richardson**

B.Sc. Artificial Intelligence & Machine Learning

GitHub: [@nathaneil-9](https://github.com/nathaneil-9)