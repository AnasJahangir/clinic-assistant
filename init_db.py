# init_db.py
from sqlmodel import Session, select
from app import create_db_and_tables, engine, load_clinic_data, Doctor

create_db_and_tables()
data = load_clinic_data()
with Session(engine) as session:
    for d in data.get('doctors', []):
        exists = session.exec(select(Doctor).where(Doctor.name == d['name'])).first()
        if not exists:
            doc = Doctor(id=d.get('id'), name=d['name'], speciality=d.get('speciality'), available=d.get('available'))
            session.add(doc)
    session.commit()

print('DB initialized and demo doctors inserted.')
