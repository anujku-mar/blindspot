# BlindSpot — Deployment Guide

## 🚀 Deploying to Streamlit Community Cloud (Recommended)

Streamlit Community Cloud is **free**, official, and designed exactly for Streamlit apps.
No code changes are needed to `app.py` or any existing files.

### Step-by-step

#### 1. Push the project to GitHub

```bash
# In the project root (Multimodal_project/)
git init
git add .
git commit -m "Initial BlindSpot project"

# Create a repo on GitHub (github.com → New Repository → "blindspot")
# Then connect and push:
git remote add origin https://github.com/YOUR_USERNAME/blindspot.git
git branch -M main
git push -u origin main
```

> **Large files:** Both `models/crosswalk_classifier.pth` (~6 MB) and
> `models/pretrained_weights/yolov8n.pt` (~6.5 MB) are under GitHub's 100 MB limit,
> so they can be committed directly. The dataset images in `dataset/crosswalk/Original/`
> (~138 MB total) may hit the limit — either use Git LFS or exclude them from the
> cloud repo (the cloud app only needs the model files to run).

If you want to exclude the dataset images to keep the repo small:
```bash
# Add to .gitignore (already included):
# dataset/crosswalk/Original/
```

#### 2. Sign up for Streamlit Community Cloud

Go to: https://share.streamlit.io
Sign in with your GitHub account (free).

#### 3. Deploy the app

1. Click **"New app"**
2. Select your repository: `YOUR_USERNAME/blindspot`
3. Branch: `main`
4. Main file path: `app.py`
5. Click **"Deploy"**

The app will build and be live at:
```
https://YOUR_USERNAME-blindspot-app-XXXX.streamlit.app
```

#### 4. Configure the app (optional)

In Streamlit Cloud dashboard → your app → **Settings → Secrets**:
No secrets are required for BlindSpot.

The `.streamlit/config.toml` file already sets the dark theme and upload limits.

---

## ☁️ Alternative: Render.com (Free Tier)

Render.com can host Streamlit apps as persistent web services (not serverless).

### Files needed (already created):
- `Dockerfile` — see below

### Dockerfile (create if deploying to Render):

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 espeak ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements_cloud.txt

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Deploy on Render: https://render.com → New Web Service → Connect GitHub repo → Done.

---

## 🌐 Alternative: Vercel (Landing Page Only)

Vercel **cannot** host Streamlit apps directly (Vercel is serverless; Streamlit needs
a persistent server). However, you can host a **landing page** on Vercel that links
to your Streamlit app.

See `vercel_landing/` folder for the ready-to-deploy landing page.

---

## 💻 Local Development (unchanged)

```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Run the app
streamlit run app.py

# Run tests
pytest -v tests/

# CLI tool
python test_single_image.py --image dataset/crosswalk/Original/c1_p1_48.jpg --speak
```

---

## 📁 Files Added for Deployment (all additive, nothing changed)

| File | Purpose |
|------|---------|
| `.gitignore` | Excludes venv, cache, secrets from git |
| `.streamlit/config.toml` | App theme and server settings |
| `packages.txt` | System packages for Streamlit Cloud (Linux) |
| `requirements_cloud.txt` | Cloud requirements (no pyttsx3 — Linux TTS) |
| `startup.py` | Model download helper for cloud (not needed locally) |
| `vercel_landing/` | Static landing page for Vercel (optional) |
| `DEPLOYMENT.md` | This file |
