<p align="center">
  <img src="https://img.shields.io/badge/DriftSense-Neural_Drift_Detection-1a1b27?style=for-the-badge&labelColor=7aa2f7&logoColor=white" alt="DriftSense"/>
</p>

<h1 align="center">🧠 DriftSense</h1>
<h3 align="center">Neural Concept Drift Detection & Localization in Process Mining</h3>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white"/></a>
  <a href="https://flask.palletsprojects.com/"><img src="https://img.shields.io/badge/Flask-3.1-000000?style=flat-square&logo=flask&logoColor=white"/></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Scikit--Learn-1.6-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/></a>
  <a href="https://ai.google.dev/"><img src="https://img.shields.io/badge/Gemini_AI-Enabled-4285F4?style=flat-square&logo=google-gemini&logoColor=white"/></a>
  <a href="https://driftsense.onrender.com"><img src="https://img.shields.io/badge/Live_Demo-Render-46E3B7?style=flat-square&logo=render&logoColor=white"/></a>
</p>

<p align="center">
  <b>An advanced process mining framework that detects and localizes Concept Drift in event logs using a Deep Reconstruction Autoencoder, powered by Explainable AI.</b>
</p>

---

## 📽️ Demo Video

<!-- 
  HOW TO ADD YOUR EXECUTION VIDEO:
  
  Option 1: Upload to YouTube (Recommended)
  - Upload your demo video to YouTube
  - Replace the link below with your YouTube URL
  
  Option 2: Upload directly to GitHub
  - Place your video file in a 'demo/' folder in this repo
  - GitHub supports .mp4 files up to 100MB
-->

<p align="center">
  <a href="https://drive.google.com/file/d/1KLM0iBOodptYUuIxZ_yZlPr4Rj6w4Zks/view?usp=sharing">
    <img src="https://img.shields.io/badge/▶_Watch_Demo-Execution_Video-FF0000?style=for-the-badge&logo=googledrive&logoColor=white" alt="Watch Demo"/>
  </a>
</p>

---

## 🌐 Live Deployment

🔗 **[https://driftsense.onrender.com](https://driftsense.onrender.com)**

---

## 🚀 Core Features

| Feature | Description |
|---------|-------------|
| 🧠 **Deep Autoencoder** | Neural network (ReLU encoder / Sigmoid decoder) learns the "grammar" of normal business processes |
| 📊 **Dynamic Thresholding** | Statistical bound (μ + 1.5σ) to detect deviations with high sensitivity |
| 🤖 **Explainable AI (XAI)** | Google Gemini API provides natural language root-cause explanations |
| 📈 **Statistical Validation** | Chi-Squared (χ²) and KL Divergence cross-verify neural detections |
| 🌐 **3D Brain Visualization** | Interactive Plotly-based latent space projection of process traces |
| 📝 **Audit Reports** | Auto-generated downloadable PDF reports with drift metrics and AI insights |
| 🎨 **Dark/Light Theme** | Cyber-glass aesthetic with smooth theme toggle |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Frontend (Browser)                  │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │ Chart.js  │  │ Plotly.js │  │ Theme / PDF Export │  │
│  └────┬─────┘  └────┬─────┘  └────────┬───────────┘  │
│       └──────────────┼────────────────┘               │
│                      │ REST API                       │
├──────────────────────┼───────────────────────────────┤
│                 Flask Backend                         │
│  ┌───────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Data Load  │→│ Autoencoder  │→│ Drift Analysis │  │
│  │ CSV / XES  │  │ (sklearn MLP)│  │ MSE / MAE     │  │
│  └───────────┘  └──────────────┘  └───────┬───────┘  │
│                                           │          │
│                      ┌────────────────────┤          │
│                      ▼                    ▼          │
│              ┌──────────────┐   ┌─────────────────┐  │
│              │ Gemini AI    │   │ Statistical     │  │
│              │ Explanations │   │ Validation      │  │
│              └──────────────┘   │ (χ², KL-Div)    │  │
│                                 └─────────────────┘  │
└──────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Layer | Technologies |
|:------|:-------------|
| **Backend** | Python 3.9+, Flask, Pandas, NumPy |
| **AI / ML** | Scikit-learn (MLPRegressor Autoencoder, PCA, MinMaxScaler) |
| **Generative AI** | Google Gemini 2.0 Flash (with multi-model fallback) |
| **Frontend** | Vanilla JS, Chart.js, Plotly.js, Custom CSS (Cyber-Glass Theme) |
| **Deployment** | Render (Gunicorn WSGI Server) |

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- A Google Gemini API Key ([Get one free](https://aistudio.google.com/apikey))

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/singh-manis/DriftSense-.git
cd DriftSense-

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
# Create a .env file in the root directory
echo GEMINI_API_KEY=your_api_key_here > .env

# 5. Run the application
python app.py
```

🌐 Open your browser at **http://127.0.0.1:5000**

---

## 📊 Methodology

The system follows a 4-phase pipeline for drift detection:

### Phase 1 — Data Ingestion
- Supports **CSV** and **XES** event log formats
- Auto-detects Case ID, Activity columns (BPI / XES compatible)
- Applies **MinMax scaling** for normalization

### Phase 2 — Neural Modeling
- Trains a **Dense Autoencoder** (MLPRegressor) to learn normal process behavior
- Architecture: `Input → ReLU Hidden Layer → Output`
- Learns compressed latent representations of process traces

### Phase 3 — Drift Detection
- Computes **Reconstruction Error** (MSE) for each trace
- Flags traces as drift candidates if: **ε_i > μ_ε + 1.5σ_ε**
- Cross-validates with **Chi-Squared (χ²)** and **KL Divergence**

### Phase 4 — Localization & Explanation
- Identifies specific **deviating features** per trace
- Generates **LLM-based root-cause analysis** via Google Gemini
- Produces downloadable **PDF audit reports**

---

## 📁 Project Structure

```
DriftSense/
├── app.py                  # Flask backend + ML pipeline
├── requirements.txt        # Python dependencies
├── Procfile                # Render deployment config
├── .env                    # API keys (not in git)
├── convert_xes.py          # XES to CSV converter utility
├── test_log.csv            # Sample test dataset
├── converted_data.csv      # Sample BPI Challenge data
│
├── static/
│   ├── script.js           # Frontend logic (charts, upload, themes)
│   └── style.css           # Cyber-glass dark theme stylesheet
│
├── templates/
│   └── index.html          # Main dashboard UI
│
└── uploads/                # Temporary file upload directory
    └── .gitkeep
```

---

## 🖥️ Dashboard Screenshots

<!-- 
  ADD YOUR SCREENSHOTS HERE:
  1. Take screenshots of your dashboard
  2. Create a 'screenshots/' folder in the repo
  3. Add images like this:
  
  ![Dashboard Overview](screenshots/dashboard.png)
  ![Drift Analysis Results](screenshots/drift_results.png)
  ![3D Brain Visualization](screenshots/brain_viz.png)
  ![AI Explanation](screenshots/ai_explanation.png)
-->

> 📌 **Add your dashboard screenshots** in a `screenshots/` folder and uncomment the image links above.

---

## 🔬 Sample Datasets

The project includes sample datasets for testing:

| Dataset | Description | Size |
|---------|-------------|------|
| `test_log.csv` | Minimal test event log | 202 bytes |
| `converted_data.csv` | Converted BPI Challenge data | 1.4 MB |

You can also download standard process mining datasets from the [BPI Challenge](https://www.tf-pm.org/resources/bpi-challenge) repository.

---

## 🚀 Deployment (Render)

This project is deployed on **[Render](https://render.com)** with the following configuration:

| Setting | Value |
|---------|-------|
| **Runtime** | Python 3 |
| **Build Command** | `pip install --no-cache-dir -r requirements.txt` |
| **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --preload` |
| **Environment Variable** | `GEMINI_API_KEY` = your API key |

---

## 👥 Team

| Name | Role |
|------|------|
| **Manish Kumar** | Project Lead & Backend Developer |
| **Mainak Patra** | Data Analyst & Documentation |
| **Ricky Mahto** | Frontend Developer & UI Designer |
| **Dr. Manoj Kumar M V** | Research Guide / Project Supervisor |

---

## 📚 References

- *"Training Neural Networks for Concept Drift Detection and Localization in Process Mining: Control-Flow Perspective"*
- [BPI Challenge Event Logs](https://www.tf-pm.org/resources/bpi-challenge)
- [Google Gemini AI Documentation](https://ai.google.dev/)

---

## 📄 License

This project is developed for academic and research purposes at **Nitte Meenakshi Institute of Technology (NMIT)**.

---

<p align="center">
  Made with 🧠 by <b>Team DriftSense</b> | NMIT, Bangalore
</p>
