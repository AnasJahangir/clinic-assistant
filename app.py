# app.py — LangChain + FastAPI clinic assistant (Groq + free embeddings)
from dotenv import load_dotenv
import os
import json
import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlmodel import SQLModel, Field, Session, create_engine, select

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq  # 👈 for Groq LLM
from langchain.prompts import PromptTemplate

# ------------------------
# Config
# ------------------------
load_dotenv()
BASE_DIR = os.path.dirname(__file__)
DB_FILE = os.path.join(BASE_DIR, "clinic.db")
CLINIC_JSON = os.path.join(BASE_DIR, "clinic_data.json")
CHROMA_DIR = os.path.join(BASE_DIR, "clinic_chroma")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------------
# Database setup
# ------------------------
class Doctor(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    speciality: Optional[str] = None
    available: Optional[str] = None

class Appointment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    patient_name: str
    contact: str
    doctor_id: Optional[int] = None
    slot: str
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

engine = create_engine(f"sqlite:///{DB_FILE}", echo=False)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# ------------------------
# LangChain AI setup (Groq + HF Embeddings)
# ------------------------
def load_clinic_data() -> dict:
    with open(CLINIC_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def build_vector_store():
    """Convert clinic_data.json into searchable vector embeddings."""
    data = load_clinic_data()
    text_data = []

    # Combine all info into text chunks
    text_data.append(f"Clinic Name: {data['name']}")
    text_data.append(f"Address: {data['address']}")
    text_data.append(f"Phone: {data['phone']}")
    text_data.append("Timings: " + json.dumps(data['timings'], indent=2))
    text_data.append("Services: " + json.dumps(data['services'], indent=2))
    text_data.append("Doctors: " + json.dumps(data['doctors'], indent=2))
    text_data.append("Notes: " + data.get("notes", ""))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = [Document(page_content=t) for t in text_data]
    split_docs = text_splitter.split_documents(docs)

    # ✅ Free embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=CHROMA_DIR)
    vectordb.persist()
    return vectordb

def get_retriever():
    if not os.path.exists(CHROMA_DIR):
        os.makedirs(CHROMA_DIR, exist_ok=True)
        build_vector_store()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 3})

# ------------------------
# FastAPI setup
# ------------------------
app = FastAPI(title="Groq Clinic Assistant")

# Initialize retriever + Groq LLM
retriever = get_retriever()
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

prompt_template = """
You are an intelligent and friendly AI assistant for {clinic_name}.
You have access to clinic details, doctor availability, and appointment timings.

Use the following context to answer accurately:
{context}

Your goals:
- Answer questions about services, timings, and doctors.
- Book appointments directly when users ask.
- Reply briefly, clearly, and naturally.

If a user says something like "Book appointment with Dr. Ayesha tomorrow 3pm",
respond like this:
"Sure! I’ve booked your appointment with Dr. Ayesha for tomorrow at 3pm."

Now answer the user's question clearly and naturally:
Question: {question}
"""

PROMPT = PromptTemplate(
    input_variables=["clinic_name", "context", "question"],
    template=prompt_template
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT.partial(clinic_name="BrightSmile Dental Clinic")},
    return_source_documents=False
)

# Demo available slots
DEMO_SLOTS = ["2025-10-15 15:00", "2025-10-15 16:00", "2025-10-16 11:00", "2025-10-16 14:00"]

def list_available_slots(session: Session) -> List[str]:
    q = session.exec(select(Appointment)).all()
    booked = {a.slot for a in q}
    return [s for s in DEMO_SLOTS if s not in booked]

def book_slot(session: Session, patient_name: str, contact: str, slot: str, doctor_id: Optional[int] = None):
    available = list_available_slots(session)
    if slot not in available:
        raise ValueError("Slot not available")
    appt = Appointment(patient_name=patient_name, contact=contact, slot=slot, doctor_id=doctor_id)
    session.add(appt)
    session.commit()
    session.refresh(appt)
    return appt

# ------------------------
# API models
# ------------------------
class ChatRequest(BaseModel):
    message: str

class BookRequest(BaseModel):
    patient_name: str
    contact: str
    slot: str
    doctor_id: Optional[int] = None

# ------------------------
# Startup initialization
# ------------------------
@app.on_event("startup")
def startup_event():
    create_db_and_tables()
    # Seed doctors if missing
    data = load_clinic_data()
    with Session(engine) as session:
        for d in data["doctors"]:
            exists = session.exec(select(Doctor).where(Doctor.name == d["name"])).first()
            if not exists:
                session.add(Doctor(**d))
        session.commit()
    print("✅ Groq Assistant ready with vector store.")

# ------------------------
# Endpoints
# ------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    try:
        answer = qa_chain.invoke({"query": req.message})
        return {"reply": answer["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/slots")
def slots():
    with Session(engine) as session:
        return {"available_slots": list_available_slots(session)}

@app.post("/book")
def book(req: BookRequest):
    with Session(engine) as session:
        try:
            appt = book_slot(session, req.patient_name, req.contact, req.slot, req.doctor_id)
            return {"status": "ok", "appointment_id": appt.id, "slot": appt.slot}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

@app.get("/doctors")
def get_doctors():
    with Session(engine) as session:
        docs = session.exec(select(Doctor)).all()
        return {"doctors": [ {"id": d.id, "name": d.name, "speciality": d.speciality, "available": d.available} for d in docs ]}
