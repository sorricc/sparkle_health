import os
from openai import AzureOpenAI
import pandas as pd
from sqlalchemy import create_engine
import json
import re
from tabulate import tabulate
from collections import defaultdict, Counter
import time
import random
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("API_KEY_PRD")
API_VERSION = '2024-08-01-preview'
deployment = 'gpt-4o-2024-11-20'
RESOURCE_ENDPOINT = os.getenv("RESOURCE_ENDPOINT_PRD")
#conn = os.getenv("SCUBA_PRD")
conn2 = os.getenv("WORKSPACE_PRD")


client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=RESOURCE_ENDPOINT,
)

#engine = create_engine(conn)
engine2 = create_engine(conn2)

# -----------------------------
# Helpers
# -----------------------------

def norm_specialties(val):
    """Normalize specialties cell into a Python list."""
    if isinstance(val, list):
        return val
    s = (val or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    return [x.strip() for x in s.split(",") if x.strip()]

def norm_term(term: str) -> str:
    """Normalize a condition term for dedupe (lowercase, remove noise)."""
    t = (term or "").strip()
    t = t.lower()

    # Remove common historical/negation markers (for safety if they slip through)
    t = re.sub(r'\b(s/p|status\s*post|h/o|hx\s*of|history\s*of|r/o|rule\s*out)\b', '', t)

    # Ophthalmology laterality shorthands
    t = re.sub(r'\b(od|os|ou|re|le|be)\b', '', t)
    t = re.sub(r'\b(of\s+both\s+eyes?|both\s+eyes?)\b', '', t)
    t = re.sub(r'\b(right|left|bilateral|rt|lt|r/l|l/r)\s+(eye|eyes)\b', '', t)
    t = re.sub(r'\b(eye|eyes)\s+(right|left|bilateral|rt|lt|r/l|l/r)\b', '', t)
    t = re.sub(r'\b(right|left|bilateral)\s+ocular\b', '', t)

    # Strip bare laterality only when clearly ocular context nearby
    t = re.sub(r'\b(right|left|bilateral|rt|lt|r/l|l/r)\b(?=.*\b(eye|eyes|ocular|retina|retinal|macula|macular|cornea|corneal|lens|optic|uvea|uveal)\b)', '', t)

    # Severity qualifiers
    t = re.sub(r'\b(severe|severely|mild|mildly|moderate|moderately)\b', '', t)
    t = re.sub(r'\b(mild[-\s]*to[-\s]*moderate|moderate[-\s]*to[-\s]*severe)\b', '', t)

    # Tiny synonym nudges
    t = re.sub(r'\b(near[-\s]*sighted(ness)?|short[-\s]*sighted(ness)?)\b', 'nearsightedness', t)

    # Tidy punctuation/spacing
    t = re.sub(r'[(),;:]', ' ', t)
    t = re.sub(r'\s*-\s*', '-', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# Few-shot examples to prime “specialty scope” + “managed vs mentioned”
FEW_SHOT_EXAMPLES = [
    {
        "Specialties": ["Orthopedics"],
        "Note": (
            "Assessment: Malunion of right distal radius fracture after prior fixation. "
            "Plan: Recommend corrective osteotomy and plate revision."
        ),
        "Output": {"conditions": [{"term": "malunion of distal radius fracture"}]}
    },
    {
        "Specialties": ["Cardiology"],
        "Note": (
            "Subjective: Hypertension well controlled at home. "
            "Assessment/Plan: New systolic heart failure; initiate guideline-directed therapy."
        ),
        "Output": {"conditions": [{"term": "systolic heart failure"}]}
    },
    {
        "Specialties": ["Dermatology"],
        "Note": (
            "HPI: Chronic plaque psoriasis with active plaques on elbows and knees. "
            "Plan: Start biologic therapy; monitor LFTs."
        ),
        "Output": {"conditions": [{"term": "plaque psoriasis"}]}
    },
    {
        "Specialties": ["Neurology"],
        "Note": (
            "Assessment: Focal epilepsy, well-controlled; continue levetiracetam. "
            "Family history positive for migraine."
        ),
        "Output": {"conditions": [{"term": "focal epilepsy"}]}
    }
]

def build_messages(specialties_list, note_text):
    """
    Build the LLM messages with stronger specialty gating, treated/managed cues,
    strict JSON guardrails, lowercase requirement, and few-shot demonstrations.
    """
    # System prompt tightened for authority boundaries + “treated” signals
    system_msg = {
        "role": "system",
        "content": (
            "You are a clinical NLP assistant. Extract only diagnosable medical conditions from the note. "
            "Prefer the most specific diagnosable child term over generic parents. Do not invent conditions. "
            "Ignore negated mentions, administrative text, family history, and side effects unless they are "
            "explicit, specialty-managed complications.\n\n"
            "Interpretation rules:\n"
            "• A condition counts only if the listed specialties are actively evaluating, diagnosing, managing, "
            "  or making plans to treat/monitor it in this note (e.g., start/adjust/continue therapy, order tests, referrals, follow-up).\n"
            "• Exclude conditions mentioned purely as history/background without a current management action.\n"
            "• Exclude general comorbidities outside the specialty’s scope (e.g., well-controlled hypertension) "
            "  unless the note explicitly indicates that the specialty is responsible for its management or it directly impacts "
            "  the specialty’s plan (e.g., perioperative risk modifications handled by the specialty).\n"
            "• If a specific child term exists, do not also list its generic parent.\n"
            "• Exclude vague symptoms (pain, cough, fatigue, diarrhea, shortness of breath) and purely historical mentions.\n\n"
            "Output rules:\n"
            "• Return only valid JSON with shape: { \"conditions\": [ { \"term\": string } ] }.\n"
            "• Use lowercase for condition terms except proper nouns.\n"
            "• If no qualifying conditions, return { \"conditions\": [] }.\n"
        )
    }

    # Few-shot demonstrations
    example_blocks = []
    for ex in FEW_SHOT_EXAMPLES:
        example_blocks.append(
            f"Example:\nSpecialties: {ex['Specialties']}\n"
            f"Note: {ex['Note']}\n"
            f"Output: {json.dumps(ex['Output'], ensure_ascii=False)}\n"
        )
    examples_text = "\n".join(example_blocks)

    user_prompt = (
        f"{examples_text}\n"
        f"Specialties: {specialties_list}\n\n"
        "Task: From the note below, extract formal conditions that are managed by these specialties. "
        "Only include conditions within the normal scope of these specialties and that are currently being "
        "evaluated, diagnosed, treated, monitored, or explicitly planned for in this note. If none qualify, return an empty list.\n\n"
        "Return JSON as:\n"
        "{ \"conditions\": [ { \"term\": string } ] }\n"
        "• term = the most specific condition name as written in the note (no invented synonyms)\n\n"
        f"Note text:\n{note_text}"
    )
    user_msg = {"role": "user", "content": user_prompt}
    return [system_msg, user_msg]

def call_llm(messages, max_retries=3, sleep_base=1.5):
    """Robust LLM call with retry + strict JSON schema."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                temperature=0.2,       # tighter, more deterministic
                top_p=1,               # standard sampling
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ExtractedConditions",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "conditions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "term": {
                                                "type": "string",
                                                "description": "lowercase condition term as written in the note; most specific form"
                                            }
                                        },
                                        "required": ["term"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["conditions"],
                            "additionalProperties": False
                        }
                    }
                }
            )
            content = response.choices[0].message.content
            # Validate JSON load
            data = json.loads(content)
            # Ensure expected keys exist
            if "conditions" not in data or not isinstance(data["conditions"], list):
                raise ValueError("Invalid JSON structure: missing 'conditions' array.")
            return data
        except Exception as e:
            if attempt == max_retries:
                raise
            # Exponential backoff with jitter
            sleep_s = sleep_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(f"LLM call error (attempt {attempt}/{max_retries}): {e}. Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)

# -----------------------------
# SQL: load notes
# -----------------------------

sql = """
SELECT top 5 n.ClinicalNoteTextKey,
       n.ProviderEpicId,
       n.ClinicalNoteEpicId,
       n.EncounterEpicCsn,
       n.Type,
       n.PrimarySpecialty,
       txt.[Text]
FROM MarketingWorkSpace.NoteExtraction.NoteConditions n
    INNER JOIN CDW.FilteredAccess.ClinicalNoteTextFact txt
        ON n.ClinicalNoteTextKey = txt.ClinicalNoteTextKey
WHERE n.RawConditionsJson IS NULL --and ProviderEpicId in ('80201')

"""

# -----------------------------
# SQL: stream notes in chunks, commit per row
# -----------------------------
from sqlalchemy import text

CHUNK_ROWS = 50

insert_sql = text("""
INSERT INTO stage.NoteConditions
(ClinicalNoteTextKey, ClinicalNoteEpicId, ProviderEpicId, EncounterEpicCsn, [Type], PrimarySpecialty, RawConditionsJson, ProcessedAt)
SELECT :ClinicalNoteTextKey, :ClinicalNoteEpicId, :ProviderEpicId, :EncounterEpicCsn, :Type, :PrimarySpecialty, :RawConditionsJson, SYSUTCDATETIME()
""")

processed = 0
errors = 0
chunk_idx = 0

# NEW: exit after 3 LLM failures
MAX_LLM_FAILURES = 3
llm_failures = 0

try:
    for df_chunk in pd.read_sql(sql, engine2, chunksize=CHUNK_ROWS):
        chunk_idx += 1
        print(f"Chunk {chunk_idx} fetched: {len(df_chunk)} rows")

        for _, row in df_chunk.iterrows():
            clinical_note_txt_key = int(row.get("ClinicalNoteTextKey")) if pd.notna(row.get("ClinicalNoteTextKey")) else None
            clinical_note_id = str(row.get("ClinicalNoteEpicId"))
            provider_id      = str(row.get("ProviderEpicId"))
            enc_csn          = int(row.get("EncounterEpicCsn")) if pd.notna(row.get("EncounterEpicCsn")) else None
            note_type        = str(row.get("Type") or "")
            primary_spec     = row.get("PrimarySpecialty", "") or ""
            note_text        = row.get("Text", "") or ""

            final_json = {"conditions": []}
            specialties_list = norm_specialties(primary_spec)

            if note_text.strip():
                try:
                    messages = build_messages(specialties_list, note_text)
                    data = call_llm(messages)
                    terms = []
                    for cond in data.get("conditions", []):
                        t = norm_term(cond.get("term", ""))
                        if t:
                            terms.append(t)
                    counts = Counter(terms)
                    final_json = {
                        "conditions": [{"term": t, "count": c} for t, c in counts.most_common()]
                    }
                except Exception as e:
                    print(f"LLM error for ClinicalNoteEpicId={clinical_note_id}: {e}")
                    # keep empty conditions so we still checkpoint the note
                    errors += 1
                    llm_failures += 1
                    if llm_failures >= MAX_LLM_FAILURES:
                        print(f"Exiting early: reached {llm_failures} LLM failures (limit {MAX_LLM_FAILURES}).")
                        raise SystemExit(1)

            # Commit this note immediately so progress is durable
            try:
                with engine2.begin() as conn_out:
                    conn_out.execute(
                        insert_sql,
                        {
                            "ClinicalNoteTextKey": clinical_note_txt_key,
                            "ClinicalNoteEpicId": clinical_note_id,
                            "ProviderEpicId": provider_id,
                            "EncounterEpicCsn": enc_csn,
                            "Type": note_type,
                            "PrimarySpecialty": primary_spec,
                            "RawConditionsJson": json.dumps(final_json, ensure_ascii=False),
                        },
                    )
                processed += 1
                print(f"Inserted note ClinicalNoteEpicId={clinical_note_id} (ProviderEpicId={provider_id}) ✅  [total {processed}]")
            except Exception as e:
                print(f"DB insert error for ClinicalNoteEpicId={clinical_note_id}: {e}")
                errors += 1

    print(f"Done. Processed {processed} notes. Errors: {errors}.")

finally:
    # Always run the stored procedure, even if exiting early due to LLM failures
    try:
        with engine2.begin() as conn2:
            print("Running stored procedure...")
            conn2.exec_driver_sql("EXEC stage.usp_FinalizeNoteConditions;")
        print("Stored procedure committed ✅")
    except Exception as e:
        print(f"Error running stored procedure: {e}")