# BlindSpot — Vercel Landing Page

This folder contains a **static landing page** for Vercel deployment.

## What this is

Vercel is a serverless/static hosting platform. It **cannot** run a Streamlit app
(which needs a persistent server process). This landing page:

- Hosts a beautiful project page on Vercel (at `https://blindspot-xyz.vercel.app`)
- Links visitors to the actual Streamlit app on Streamlit Community Cloud
- Does NOT touch any project code

## How to deploy this landing page on Vercel

### Option A: Vercel CLI

```bash
# Install Vercel CLI (one-time)
npm install -g vercel

# Navigate to this folder
cd vercel_landing/

# Deploy
vercel --prod
```

### Option B: Vercel Dashboard (no CLI)

1. Go to https://vercel.com and sign up / log in with GitHub
2. Click "Add New Project"
3. Import your GitHub repo → set **Root Directory** to `vercel_landing/`
4. Click Deploy

Your landing page will be live at something like:
`https://blindspot-abc123.vercel.app`

## After deploying Streamlit Community Cloud

Once your Streamlit app is live at `https://YOUR_APP_URL.streamlit.app`:

1. Open `vercel_landing/index.html`
2. Replace all occurrences of `YOUR_APP_URL` with your actual Streamlit URL
3. Replace `YOUR_USERNAME` with your GitHub username
4. Re-deploy (`vercel --prod`)

## Important

The main Streamlit app is in the parent directory and runs with:
```bash
streamlit run app.py
```
Nothing in this folder touches that.
