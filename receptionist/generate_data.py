import json
import os
import random
from faker import Faker

fake = Faker()

# Create output folder
os.makedirs("dummy_patients", exist_ok=True)

diagnoses = [
    "Chronic Kidney Disease Stage 3",
    "Acute Kidney Injury",
    "Nephrotic Syndrome",
    "Hypertensive Nephropathy",
    "Diabetic Nephropathy",
    "End-Stage Renal Disease",
    "Polycystic Kidney Disease",
    "Glomerulonephritis",
    "Renal Tubular Acidosis",
    "Chronic Pyelonephritis"
]

medications_list = [
    ["Lisinopril 10mg daily", "Furosemide 20mg twice daily"],
    ["Amlodipine 5mg daily", "Sodium Bicarbonate 500mg thrice daily"],
    ["Losartan 25mg daily", "Hydrochlorothiazide 12.5mg daily"],
    ["Erythropoietin 4000 IU weekly", "Calcium Carbonate 500mg twice daily"],
    ["Metformin 500mg twice daily", "Atorvastatin 10mg daily"]
]

dietary_restrictions = [
    "Low sodium (2g/day), fluid restriction (1.5L/day)",
    "Low potassium diet, protein restriction (0.8g/kg/day)",
    "Low phosphorus diet, fluid restriction (2L/day)",
    "Renal diet, avoid high-potassium fruits",
    "Low salt, no processed food, fluid restriction (1.2L/day)"
]

warning_signs = [
    "Swelling, shortness of breath, decreased urine output",
    "Chest pain, dizziness, rapid weight gain",
    "Persistent nausea, confusion, swelling in legs",
    "Fever, chills, pain while urinating",
    "Severe fatigue, itching, loss of appetite"
]

instructions = [
    "Monitor blood pressure daily, weigh yourself daily",
    "Take medications on time, report any swelling or dizziness",
    "Avoid high-salt foods, keep track of urine output",
    "Follow up regularly with nephrology clinic",
    "Stay hydrated within fluid limits, track your symptoms"
]

# Generate 25+ patients
for i in range(1, 30):
    patient = {
        "patient_name": fake.name(),
        "discharge_date": str(fake.date_between(start_date='-6M', end_date='today')),
        "primary_diagnosis": random.choice(diagnoses),
        "medications": random.choice(medications_list),
        "dietary_restrictions": random.choice(dietary_restrictions),
        "follow_up": "Nephrology clinic in 2 weeks",
        "warning_signs": random.choice(warning_signs),
        "discharge_instructions": random.choice(instructions)
    }

    with open(f"dummy_patients/patient_{i}.json", "w") as f:
        json.dump(patient, f, indent=4)

print("Generated 30 dummy nephrology patient reports in /dummy_patients/")
