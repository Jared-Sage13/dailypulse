# DailyPulse (public)

Live markets, sectors, screener, calendar, news, and a computed daily digest.
No API keys required. Flask app served by gunicorn.

## Run locally
```
pip install -r requirements.txt
python3 app.py            # http://localhost:5055
```

## Deploy (Render, free)
This repo includes `render.yaml` (Blueprint) and a `Procfile`.
1. Push this folder to a GitHub repo.
2. On Render: New → Blueprint → connect the repo → Apply.
   (Or New → Web Service → build `pip install -r requirements.txt`,
    start `gunicorn app:app --bind 0.0.0.0:$PORT`.)
