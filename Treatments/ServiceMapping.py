import os
import json
import time
import random
import re
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from openai import AzureOpenAI

load_dotenv()

# -----------------------------
# Config
# -----------------------------
API_KEY = os.getenv("API_KEY_PRD")
API_VERSION = "2024-08-01-preview"
deployment = "gpt-4o-2024-08-06"
RESOURCE_ENDPOINT = os.getenv("RESOURCE_ENDPOINT_PRD")

conn2 = os.getenv("WORKSPACE_PRD")  # where NoteExtraction.CptHcpcsServiceMap lives
engine2 = create_engine(conn2)

client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=RESOURCE_ENDPOINT,
)

CHUNK_ROWS = 200

# -----------------------------
# Helpers
# -----------------------------
def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_label(s: str) -> str:
    """Conservative normalization for display/group labels."""
    s = clean_text(s)
    s = re.sub(r"\bIV\b", "IV", s, flags=re.IGNORECASE)
    s = re.sub(r"\bMRI\b", "MRI", s, flags=re.IGNORECASE)
    s = re.sub(r"\bCT\b", "CT", s, flags=re.IGNORECASE)
    s = re.sub(r"\bECG\b", "ECG", s, flags=re.IGNORECASE)
    s = re.sub(r"\bEKG\b", "ECG", s, flags=re.IGNORECASE)
    return s

def safe_load_json(maybe_json: str):
    try:
        return json.loads(maybe_json) if maybe_json else None
    except Exception:
        return None

def yn(val: str) -> str:
    """Normalize Y/N outputs defensively."""
    v = (val or "").strip().upper()
    return "Y" if v == "Y" else "N"

# -----------------------------
# Prompt builders (minimal changes; add new flags)
# -----------------------------
def build_messages(code, code_type, procedure_detail, descriptions_json):
    system_msg = {
        "role": "system",
        "content": (
            "You are a healthcare marketing taxonomy assistant. "
            "Your job is to map CPT/HCPCS procedure codes to consumer-facing services for provider profiles.\n\n"
            "Principles:\n"
            "• Be consumer-friendly and SEO-safe.\n"
            "• Ignore billing minutiae: time increments (15/30/60 minutes), 'initial', 'each additional', "
            "  'with/without', training, and technical qualifiers unless they materially change the service.\n"
            "• Prefer stable, reusable grouping names.\n"
            "• Do NOT invent capabilities; base decisions on the provided descriptions.\n"
            "• Do NOT judge eligibility based on how common, rare, highly specialized, or technical a procedure is.\n\n"
            "Output rules:\n"
            "• Return only valid JSON matching the schema.\n"
            "• display_label: 2–4 words, Title Case.\n"
            "• service_group: 2–5 words, Title Case; may equal display_label.\n"
            "• display_eligible: 'Y' for clinically valid services that may appear on a provider profile, "
            "  even if rare or highly specialized; "
            "  'N' only for administrative, ambiguous, billing-only, or unlisted/catch-all procedure codes.\n"
            "• is_surgical/is_imaging/is_therapy: classify based on clinical meaning (not billing).\n"
            "• surgical_type: if is_surgical='Y', choose one of: Open, Laparoscopic, Robotic, Endoscopic, Percutaneous, Other. "
            "  If not surgical, use null.\n"
        ),
    }

    user_payload = {
        "billing_code": code,
        "code_type": code_type,
        "procedure_detail": procedure_detail,
        "description_variants": safe_load_json(descriptions_json) or [],
    }

    user_msg = {
        "role": "user",
        "content": (
            "Map this procedure code to a consumer-facing service.\n\n"
            "Return JSON with:\n"
            "{\n"
            '  "service_group": string,\n'
            '  "display_label": string,\n'
            '  "display_eligible": "Y"|"N",\n'
            '  "exclude_reason": string|null,\n'
            '  "notes": string|null,\n'
            '  "is_surgical": "Y"|"N",\n'
            '  "surgical_type": "Open"|"Laparoscopic"|"Robotic"|"Endoscopic"|"Percutaneous"|"Other"|null,\n'
            '  "is_imaging": "Y"|"N",\n'
            '  "is_therapy": "Y"|"N"\n'
            "}\n\n"
            "Guidance on display_eligible:\n"
            "• 'Y' for clinically valid services that may appear on a provider profile, even if rare or highly specialized.\n"
            "• 'N' only for unlisted/unspecified catch-all codes or administrative/billing-only concepts.\n"
            "• Do NOT exclude a service because it is rare, complex, or highly specialized.\n"
            "• If 'N', set exclude_reason to a brief plain-English reason.\n\n"
            "Guidance on flags:\n"
            "• is_surgical='Y' for operative/procedural surgical interventions (including endoscopic/robotic/percutaneous when appropriate).\n"
            "• is_imaging='Y' for diagnostic imaging services (e.g., X-ray, CT, MRI, PET, ultrasound, fluoroscopy when used as imaging).\n"
            "• is_therapy='Y' for rehab/therapy services (PT/OT/SLP/behavioral therapy).\n"
            "• surgical_type: set only when is_surgical='Y'; otherwise null.\n\n"
            f"Input:\n{json.dumps(user_payload, ensure_ascii=False)}"
        ),
    }

    return [system_msg, user_msg]

# -----------------------------
# LLM call (updated schema)
# -----------------------------
def call_llm(messages, max_retries=4, sleep_base=1.5):
    schema = {
        "name": "ServiceMap",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "service_group": {"type": "string"},
                "display_label": {"type": "string"},
                "display_eligible": {"type": "string", "enum": ["Y", "N"]},
                "exclude_reason": {"type": ["string", "null"]},
                "notes": {"type": ["string", "null"]},
                "is_surgical": {"type": "string", "enum": ["Y", "N"]},
                "surgical_type": {
                    "type": ["string", "null"],
                    "enum": ["Open", "Laparoscopic", "Robotic", "Endoscopic", "Percutaneous", "Other", None],
                },
                "is_imaging": {"type": "string", "enum": ["Y", "N"]},
                "is_therapy": {"type": "string", "enum": ["Y", "N"]},
            },
            "required": [
                "service_group",
                "display_label",
                "display_eligible",
                "exclude_reason",
                "notes",
                "is_surgical",
                "surgical_type",
                "is_imaging",
                "is_therapy",
            ],
            "additionalProperties": False,
        },
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=messages,
                temperature=0.2,
                top_p=1,
                response_format={"type": "json_schema", "json_schema": schema},
            )
            content = resp.choices[0].message.content
            data = json.loads(content)

            # normalize + defensive defaults
            data["service_group"] = norm_label(data.get("service_group", "Unmapped"))
            data["display_label"] = norm_label(data.get("display_label", "Unmapped"))
            data["display_eligible"] = yn(data.get("display_eligible", "N"))
            data["is_surgical"] = yn(data.get("is_surgical", "N"))
            data["is_imaging"] = yn(data.get("is_imaging", "N"))
            data["is_therapy"] = yn(data.get("is_therapy", "N"))

            # surgical_type only valid if is_surgical=Y
            if data["is_surgical"] != "Y":
                data["surgical_type"] = None
            else:
                st = (data.get("surgical_type") or "").strip()
                allowed = {"Open", "Laparoscopic", "Robotic", "Endoscopic", "Percutaneous", "Other"}
                data["surgical_type"] = st if st in allowed else "Other"

            if data["display_eligible"] == "N" and not data.get("exclude_reason"):
                data["exclude_reason"] = "Not appropriate for public provider service list."

            return data, content
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                raise
            sleep_s = sleep_base * (2 ** (attempt - 1)) + random.uniform(0, 0.6)
            print(f"LLM error (attempt {attempt}/{max_retries}): {e}. Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)

    raise last_err

# -----------------------------
# SQL: select distinct codes to map (skip already mapped)
# -----------------------------
sql = """
SELECT DISTINCT 
       c.BillingCode,
       c.CodeType,
       c.ProcedureDetail,
       c.DescriptionsJson
FROM MarketingWorkSpace.NoteExtraction.ProviderTreatmentCounts c
LEFT JOIN NoteExtraction.CptHcpcsServiceMap m
  ON m.BillingCode = c.BillingCode
 AND m.CodeType    = c.CodeType
WHERE m.BillingCode IS NULL;
"""

# -----------------------------
# SQL: per-row upsert (updated for new columns)
# -----------------------------
upsert_sql = text("""
MERGE NoteExtraction.CptHcpcsServiceMap AS tgt
USING (SELECT
    :BillingCode AS BillingCode,
    :CodeType    AS CodeType
) AS src
ON (tgt.BillingCode = src.BillingCode AND tgt.CodeType = src.CodeType)
WHEN MATCHED THEN
  UPDATE SET
    ProcedureDetail  = :ProcedureDetail,
    DescriptionsJson = :DescriptionsJson,
    ServiceGroup     = :ServiceGroup,
    DisplayLabel     = :DisplayLabel,
    DisplayEligible  = :DisplayEligible,
    ExcludeReason    = :ExcludeReason,
    Notes            = :Notes,
    IsSurgical       = :IsSurgical,
    SurgicalType     = :SurgicalType,
    IsImaging        = :IsImaging,
    IsTherapy        = :IsTherapy,
    RawModelJson     = :RawModelJson,
    ModelName        = :ModelName,
    ModelVersion     = :ModelVersion,
    GeneratedAtUtc   = SYSUTCDATETIME()
WHEN NOT MATCHED THEN
  INSERT (
    BillingCode, CodeType, ProcedureDetail, DescriptionsJson,
    ServiceGroup, DisplayLabel, DisplayEligible, ExcludeReason, Notes,
    IsSurgical, SurgicalType, IsImaging, IsTherapy,
    RawModelJson, ModelName, ModelVersion, GeneratedAtUtc
  )
  VALUES (
    :BillingCode, :CodeType, :ProcedureDetail, :DescriptionsJson,
    :ServiceGroup, :DisplayLabel, :DisplayEligible, :ExcludeReason, :Notes,
    :IsSurgical, :SurgicalType, :IsImaging, :IsTherapy,
    :RawModelJson, :ModelName, :ModelVersion, SYSUTCDATETIME()
  );
""")

processed = 0
errors = 0
chunk_idx = 0

for df_chunk in pd.read_sql(sql, engine2, chunksize=CHUNK_ROWS):
    chunk_idx += 1
    print(f"Chunk {chunk_idx} fetched: {len(df_chunk)} codes")

    for _, row in df_chunk.iterrows():
        billing_code     = clean_text(str(row.get("BillingCode") or ""))
        code_type        = clean_text(str(row.get("CodeType") or ""))
        procedure_detail = clean_text(str(row.get("ProcedureDetail") or ""))
        desc_json        = str(row.get("DescriptionsJson") or "[]")

        if not billing_code or not code_type:
            continue

        # Default placeholder (in case LLM fails)
        mapped = {
            "service_group": "Unmapped",
            "display_label": "Unmapped",
            "display_eligible": "N",
            "exclude_reason": "LLM failure or missing description.",
            "notes": None,
            "is_surgical": "N",
            "surgical_type": None,
            "is_imaging": "N",
            "is_therapy": "N",
        }
        raw_model_json = json.dumps(mapped, ensure_ascii=False)

        try:
            messages = build_messages(billing_code, code_type, procedure_detail, desc_json)
            mapped, raw_model_json = call_llm(messages)
        except Exception as e:
            print(f"LLM error for code={billing_code} ({code_type}): {e}")
            errors += 1

        # Commit per code immediately (durable progress)
        try:
            with engine2.begin() as conn_out:
                conn_out.execute(
                    upsert_sql,
                    {
                        "BillingCode": billing_code,
                        "CodeType": code_type,
                        "ProcedureDetail": procedure_detail if procedure_detail else None,
                        "DescriptionsJson": desc_json,

                        "ServiceGroup": mapped["service_group"],
                        "DisplayLabel": mapped["display_label"],
                        "DisplayEligible": mapped["display_eligible"],
                        "ExcludeReason": mapped.get("exclude_reason"),
                        "Notes": mapped.get("notes"),

                        "IsSurgical": mapped.get("is_surgical", "N"),
                        "SurgicalType": mapped.get("surgical_type"),
                        "IsImaging": mapped.get("is_imaging", "N"),
                        "IsTherapy": mapped.get("is_therapy", "N"),

                        "RawModelJson": raw_model_json,
                        "ModelName": deployment,
                        "ModelVersion": API_VERSION,
                    },
                )

            processed += 1
            print(f"Upserted {billing_code} ({code_type}) ✅ [total {processed}]")
        except Exception as e:
            print(f"DB upsert error for code={billing_code} ({code_type}): {e}")
            errors += 1

print(f"Done. Processed {processed} codes. Errors: {errors}.")