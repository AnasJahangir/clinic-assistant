Clinic AI Assistant - Demo (single-file FastAPI)
==============================================

Files (copy these files into a folder):
 - app.py
 - clinic_data.json
 - init_db.py
 - requirements.txt

Quick start (on your local machine):
1) Create & activate venv:
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows (PowerShell)

2) Install:
   pip install -r requirements.txt

3) Initialize DB:
   python init_db.py

4) Run server:
   uvicorn app:app --reload --host 0.0.0.0 --port 8000

API examples:
 - GET /doctors
   curl http://127.0.0.1:8000/doctors

 - GET /slots
   curl http://127.0.0.1:8000/slots

 - POST /chat (info)
   curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"message":"Root canal ka charge kya hai?"}'

 - POST /book
   curl -X POST http://127.0.0.1:8000/book -H "Content-Type: application/json" -d '{"patient_name":"Ali","contact":"+92-300-1112223","slot":"2025-10-15 15:00"}'

OpenAI integration:
 - To use GPT-style replies set environment variable OPENAI_API_KEY before starting the server.
 - app.py uses the OpenAI Python SDK (client named `OpenAI`) and calls model "gpt-4o-mini" in the example.
