import os
import json
import pandas as pd


DATA_DIR = "dummy_patients"
OUTPUT_CSV = os.path.join(DATA_DIR, "patient_index.csv")

def generate_patient_index(data_dir=DATA_DIR, output_csv=OUTPUT_CSV):
    rows = []

   
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(data_dir, file_name)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                
                patient_name = data.get("patient_name", "Unknown")

                rows.append({
                    "patient_name": patient_name,
                    "file_name": file_name
                })
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"✅ Patient index CSV generated successfully at: {output_csv}")
    print(f"📄 Total patients indexed: {len(df)}")
    print(df.head())

if __name__ == "__main__":
    generate_patient_index()
