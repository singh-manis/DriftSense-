# DriftSense: Neural Concept Drift Detection & Localization

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Gemini](https://img.shields.io/badge/Gemini_AI-Enabled-blue?style=flat-square&logo=google-gemini&logoColor=white)](https://ai.google.dev/)

**DriftSense** is an advanced process mining framework designed to detect and localize **Concept Drift** in event logs using a Deep Reconstruction Autoencoder architecture. This project implements the methodology described in the research paper: *"Training Neural Networks for Concept Drift Detection and Localization in Process Mining: Control-Flow Perspective"*.

---

## 🚀 Core Features

-   **Deep Reconstruction Autoencoder**: Utilizes a neural network (ReLU Encoder / Sigmoid Decoder) to learn the "grammar" of normal business processes.
-   **Dynamic Drift Thresholding**: Implements a statistical bound $(\mu + 1.5\sigma)$ to detect deviations with high sensitivity.
-   **Explainable AI (XAI)**: Integrated with **Google Gemini API** to provide natural language explanations for detected anomalies.
-   **Statistical Validation**: Cross-verifies neural detections using **Chi-Squared ($\chi^2$)** and **Kullback-Leibler (KL) Divergence** metrics.
-   **3D Latent Space Visualization**: Interactive Plotly-based "Brain Visualization" to project high-dimensional process traces into a navigable 3D manifold.
-   **Automated Audit Reporting**: Generates downloadable PDF reports (via jsPDF) containing drift metrics and AI insights.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend** | Python, Flask, PM4Py, Pandas, NumPy |
| **AI/ML** | TensorFlow, Keras (Deep Autoencoder), Scikit-learn (PCA, Scaling) |
| **Generative AI** | Google Gemini Generative AI SDK |
| **Frontend** | Vanilla JS, Chart.js, Plotly.js, Vis.js (CFG Graphs), Tailwind-inspired CSS |

---

## 📦 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/MainakPatra290803/Final-Year-Project.git
    cd Final-Year-Project
    ```

2.  **Install Dependencies** (Recommended: Python 3.9+)
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    Create a `.env` file in the root directory and add your Gemini API Key:
    ```env
    GEMINI_API_KEY=your_google_gemini_api_key_here
    ```

4.  **Run the Application**
    ```bash
    python app.py
    ```
    Access the dashboard at `http://127.0.0.1:5000/`.

---

## 📊 Methodology (Page-by-Page Audit)

-   **Phase 1: Ingestion**: Supports XES and CSV event log formats with automated MinMax scaling.
-   **Phase 2: Modeling**: Trains a Dense Autoencoder to minimize Mean Squared Error (MSE).
-   **Phase 3: Detection**: Flags traces as drift candidates if $\epsilon_i > \mu_{\epsilon} + 1.5\sigma_{\epsilon}$.
-   **Phase 4: Localization**: Identifies specific deviating features and generates LLM-based root-cause analysis.

---

## 👥 Meet the Team

-   **Mainak Patra** (Data Analyst & Documentation)
-   **Ricky Mahto** (Frontend Developer & UI Designer)
-   **Manish Kumar** (Project Lead & Backend Developer)
-   **Dr. Manoj Kumar M V** (Research Guide / Project Supervisor)

---

## 📄 License
This project is developed for research purposes at **Nitte Meenakshi Institute of Technology (NMIT)**.
