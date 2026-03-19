import json
import re
import time
import hashlib
#import pandas as pdimport os
from openai import AzureOpenAI
import pandas as pd
from sqlalchemy import create_engine
import json
import re
from tabulate import tabulate 
from collections import defaultdict, Counter
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv
import os
import ast



load_dotenv() 

API_KEY = os.getenv("API_KEY_PRD")
API_VERSION = '2024-08-01-preview'
deployment = 'gpt-5-2025-08-07'
RESOURCE_ENDPOINT = os.getenv("RESOURCE_ENDPOINT_PRD")
conn = os.getenv("SCUBA_PRD")
conn2 = os.getenv("SCUBA_DEV")

client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=RESOURCE_ENDPOINT,
)



engine2 = create_engine(conn2)

# Where to store results
LLM_TARGET_TABLE = "ConditionConsumerTermsV2"   

# ---- Helper: simple dedupe/quality gates for terms ----
def _norm_term(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*[-–—]\s*", "-", s)
    return s


def _is_bad_term(t: str, pn_norm: str) -> bool:
    # Filter out trivial/low-quality outputs
    if not t:
        return True
    if len(t) > 60:  # too long = likely phrase/sentence
        return True
    if t == pn_norm:  # identical to preferred name
        return True
    if re.fullmatch(r"[a-z]{1,3}", t):  # acronyms like "dm" / "htn"
        return True
    if re.search(r"\b(icd|snomed|c\d{7})\b", t):  # codes/ids
        return True
    if re.search(r"\d{2,}", t):  # numeric-y
        return True
    if re.search(r"[{}[\]|<>]", t):
        return True
    return False




SYSTEM_PROMPT = (
    "You are a medical plain-language and colloquial mapper. "
    "Given a clinical condition (Preferred Name), produce up to 20 U.S. English consumer-facing terms "
    "that real people might say, including common phrasing and safe colloquialisms. "
    "Avoid clinical jargon, abbreviations, and codes. Be respectful and non-stigmatizing. "
    "Always respond using the provided JSON schema only—no prose."
)

FEW_SHOT_USER = (
    "Preferred name: Diabetes Mellitus\n"
    "Generate up to 20 consumer-facing terms people use in everyday speech.\n"
    "Keep each term concise—typically 1–4 words."
)

FEW_SHOT_ASSISTANT = json.dumps({
    "cui": "C0011849",
    "preferred_name": "Diabetes Mellitus",
    "terms": [
        {"term": "diabetes", "register": "common"},
        {"term": "high blood sugar", "register": "common"},
        {"term": "sugar diabetes", "register": "colloquial"},
        {"term": "the sugars", "register": "slang"}
    ]
}, ensure_ascii=False)




LLM_SCHEMA = {
    "name": "ConsumerTerms",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "cui": {"type": "string"},
            "preferred_name": {"type": "string"},
            "terms": {
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string"},
                        "register": {"type": "string", "enum": ["common", "colloquial", "slang"]}
                    },
                    "required": ["term", "register"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["cui", "preferred_name", "terms"],
        "additionalProperties": False
    }
}


def _prompt_messages(cui: str, preferred_name: str):
    user_task = (
        "Task:\n"
        f"- Preferred name: {preferred_name}\n"
        "- Generate up to 20 consumer-facing terms people use in everyday speech.\n"
        "- Keep each term 1–4 words. Lowercase is fine.\n"
        "- Include truly plain phrasing and safe colloquialisms.\n"
        "- Exclude clinical jargon, abbreviations, brand names, codes, and overly technical phrasing.\n"
        "- Do not include prose explanations; return only the JSON object per the schema.\n"
        f"- Set 'cui' to '{cui}'."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
        {"role": "user", "content": user_task}
    ]


# =========================
# LLM Generation
# =========================

def generate_consumer_terms_llm(cui: str, preferred_name: str, max_retries: int = 2):
    # Compose prompt + hash for lineage
    prompt_json = {"cui": cui, "preferred_name": preferred_name}
    prompt_hash = hashlib.sha256(json.dumps(prompt_json, sort_keys=True).encode()).hexdigest()[:16]

    messages = _prompt_messages(cui, preferred_name)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=deployment,
                temperature=1,  
                presence_penalty=0.0,
                response_format={"type": "json_schema", "json_schema": LLM_SCHEMA},
                messages=messages,
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)

            # Post-process quality gates
            pn_norm = _norm_term(preferred_name)
            seen = set()
            cleaned = []
            for item in data.get("terms", []):
                term = _norm_term(item.get("term", ""))
                if _is_bad_term(term, pn_norm):
                    continue
                if term in seen:
                    continue
                seen.add(term)
                cleaned.append({
                    "term": term,
                    "register": item.get("register", "common")
                })
                if len(cleaned) >= 20:
                    break

            out = {
                "cui": cui,
                "preferred_name": preferred_name,
                "terms": cleaned
            }
            return out, prompt_hash

        except Exception as e:
            last_err = e
            time.sleep(0.5 * attempt)

    raise RuntimeError(f"LLM generation failed for {cui}: {last_err}")



sql = '''
WITH f AS
(
    SELECT DISTINCT
           j.[key] COLLATE SQL_Latin1_General_CP1_CI_AS          AS cui,
           JSON_VALUE(j.value, '$.canonical_name')
                COLLATE SQL_Latin1_General_CP1_CI_AS             AS PreferredName
    FROM NoteExtraction.NoteConditions AS n
    CROSS APPLY OPENJSON(n.MappedConditions) AS j
    WHERE n.MappedConditions IS NOT NULL
)
SELECT f.cui, f.PreferredName
FROM f
WHERE NOT EXISTS
(
    SELECT 1
    FROM NoteExtraction.ConditionConsumerTermsV2 AS cct
    WHERE cct.cui = f.cui
);


'''
df_pn = pd.read_sql(sql, con=engine2)

for _, r in df_pn.iterrows():
    cui = str(r["cui"])
    pn  = str(r["PreferredName"])

    try:
        result, phash = generate_consumer_terms_llm(cui, pn)

        # one-row DataFrame -> append straight into SQL
        df = pd.DataFrame([{
            "cui": cui,
            "TermsJson": json.dumps(result, ensure_ascii=False),
            "model": deployment,
            "prompt_hash": phash
        }])
        #extract items from TermsJson
        df['TermsJson'] = df['TermsJson'].to_dict()
        df['TermsJson'] = df['TermsJson'].apply(lambda x:ast.literal_eval(x))
        df['preferred_name'] = df['TermsJson'].apply(lambda x:x['preferred_name'])
        df['idx_cui(cui)'] = df['TermsJson'].apply(lambda x:x['cui'])
        df['terms'] = df['TermsJson'].apply(lambda x:x['terms'])
        df['TermsJson'] = df['TermsJson'].apply(lambda x:json.dumps(x))
        df['terms'] = df['terms'].apply(lambda x:json.dumps(x))
        df.rename(columns={'TermsJson':'idx_terms_fulltext(preferred_name)',
                        'Id':'id',
                        'CreatedAt': 'created_at'},
                        inplace=True)
        df.to_sql(
            LLM_TARGET_TABLE,
            con=engine2,
            schema='NoteExtraction',
            if_exists="append",
            index=False
        )

        print(f"✓ {cui} - {pn}: {len(result['terms'])} terms (inserted)")

    except IntegrityError:
        # if you kept the UNIQUE (CUI, PromptHash) index
        print(f"⤴︎ duplicate skipped for {cui} ({pn})")

    except Exception as e:
        print(f"⚠️ {cui} - {pn}: {e}")