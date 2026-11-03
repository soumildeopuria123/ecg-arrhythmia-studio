nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload >backend.log 2>&1 &
