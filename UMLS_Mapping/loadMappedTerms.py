import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text, event
from sqlalchemy.dialects.mssql import NVARCHAR
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = r"C:\Users\sorricc\Documents\GitHub\SparkleHealthNew\UMLS_Mapping\data\Note_terms_output.csv"
if not CSV_PATH:
    print("Provide path to CSV: python insert_note_conditions_stage.py <csv_path>")
    sys.exit(1)
    

CONN = os.getenv("WORKSPACE_PRD")
if not CONN:
    print("Set WORKSPACE_PRD environment variable with a SQLAlchemy connection string.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH, usecols=["ClinicalNoteEpicId", "MappedByCui", "UnmappedConditions"])
df["ClinicalNoteEpicId"] = df["ClinicalNoteEpicId"].astype(str).str.strip()
df = df[df["ClinicalNoteEpicId"] != ""]

engine = create_engine(
    CONN,
    connect_args={"fast_executemany": True}
)



with engine.begin() as conn:
    # If you want pandas to create columns with specific types on first run:
    dtype_map = {
        "ClinicalNoteEpicId": NVARCHAR(length=100),
        "MappedByCui": NVARCHAR(None),           # NVARCHAR(MAX)
        "UnmappedConditions": NVARCHAR(None),    # NVARCHAR(MAX)
    }
    df.to_sql(
        "NoteMappedConditions",
        schema="stage",
        con=conn,
        if_exists="append",
        index=False,
        dtype=dtype_map,
        chunksize=10_000
    )

print(f"Done. Inserted {len(df)} row(s) into stage.NoteMappedConditions.")

try:
    with engine.begin() as conn:   
        print("Running stored procedure...")
        conn.exec_driver_sql("EXEC stage.usp_FinalizeNoteMapping;")
    print("Stored procedure committed ✅")
except Exception as e:
    print(f"Error running stored procedure: {e}")
