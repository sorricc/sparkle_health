import os, json
import pandas as pd
from sqlalchemy import create_engine, text
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import List, Union

# -------------------------
# Environment / Clients
# -------------------------
load_dotenv()
API_KEY = os.getenv("API_KEY_PRD")
RESOURCE_ENDPOINT = os.getenv("RESOURCE_ENDPOINT_PRD")
API_VERSION = "2024-08-01-preview"
DEPLOYMENT = "gpt-5-2025-08-07"

SCUBA_DEV = os.getenv("SCUBA_DEV")  # same DSN/URL you’re already using
engine = create_engine(SCUBA_DEV)

client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=RESOURCE_ENDPOINT,
)

# -------------------------
# SQL queries
# -------------------------
# Providers to process (distinct), with their ApexPrimarySpeciality
SQL_PROVIDER_LIST = """
SELECT DISTINCT
       pc.ProviderEpicId,
       pc.ApexPrimarySpeciality
FROM NoteExtraction.ProviderConditionsFinal pc
WHERE pc.MappedConditions IS NOT NULL 
"""

# Evidence rows (CUIs + names + counts) per provider
SQL_PROVIDER_EVIDENCE = """
SELECT f.cui,
       f.PreferredName,
       f.TotalCount,
       f.ApexPrimarySpeciality,
       f.ProviderEpicId
FROM (
    SELECT DISTINCT 
           JSON_VALUE(j.value, '$.cui')              AS cui,
           JSON_VALUE(j.value, '$.canonical_name')   AS PreferredName,
           CAST(JSON_VALUE(j.value, '$.total_count') AS INT) AS TotalCount,
           pc.ApexPrimarySpeciality,
           pc.ProviderEpicId
    FROM NoteExtraction.ProviderConditionsFinal  pc
    CROSS APPLY OPENJSON(pc.MappedConditions) j
    WHERE TRY_CAST(JSON_VALUE(j.value, '$.total_count') AS INT) > 1
    and ApexPrimarySpeciality is not NULL
    AND pc.NoteCount > 50
) f
WHERE f.ProviderEpicId = :provider_id
"""

# Allowed subspecialties by primary specialty
SQL_ALLOWED_SUBS = """
SELECT SubSpeciality
FROM NoteExtraction.SubSpecialityMapping
WHERE PrimarySpeciality = :primary_specialty
ORDER BY SubSpeciality
"""

# Update the three subspecialty columns for a provider
SQL_UPDATE_SUBS = """
UPDATE pc
SET
    DerivedSubSpeciality1 = :s1,
    DerivedSubSpeciality2 = :s2,
    DerivedSubSpeciality3 = :s3
FROM NoteExtraction.ProviderConditionsFinal pc
WHERE pc.ProviderEpicId = :provider_id
"""

# -------------------------
# LLM prompt + schema
# -------------------------
SYSTEM_PROMPT = (
    "You are a clinical classification assistant. "
    "Given a provider’s primary specialty and top condition evidence (CUIs with names and counts), "
    "choose up to THREE subspecialties ONLY from the allowed list. "
    "If evidence is insufficient, choose none. Be conservative for scheduling. "
    "Return JSON only."
)

SUBSPEC_SCHEMA = {
    "name": "DerivedSubspecialties",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "ProviderEpicId": {"type": "string"},
            "DerivedSubSpeciality1": {"type": ["string", "null"]},
            "DerivedSubSpeciality2": {"type": ["string", "null"]},
            "DerivedSubSpeciality3": {"type": ["string", "null"]}
        },
        "required": [
            "ProviderEpicId",
            "DerivedSubSpeciality1",
            "DerivedSubSpeciality2",
            "DerivedSubSpeciality3"
        ],
        "additionalProperties": False
    }
}


def _build_user_prompt(primary_specialty: str, allowed_subs: List[str], evidence_rows: List[dict], provider_id: str) -> str:
    # Keep top 20 by count for signal
    rows = sorted(evidence_rows, key=lambda r: int(r.get("TotalCount", 0)), reverse=True)[:20]
    lines = [
        f"Primary specialty: {primary_specialty}",
        "Allowed subspecialties (choose up to 3; choose none if unclear):",
    ]
    lines += [f"- {s}" for s in allowed_subs]
    lines.append("")
    lines.append("Evidence (top conditions):")
    for r in rows:
        lines.append(f"- {r['PreferredName']} (CUI {r['cui']}): count={r['TotalCount']}")
    lines += [
        "",
        "Task:",
        "Select up to three subspecialties from the allowed list that best match the evidence.",
        "If none are clearly supported, return only the ProviderEpicId with no subspecialty fields.",
        f"ProviderEpicId: {provider_id}"
    ]
    return "\n".join(lines)

def _classify_one(provider_id: str) -> dict:
    # Pull evidence
    df_ev = pd.read_sql(text(SQL_PROVIDER_EVIDENCE), con=engine, params={"provider_id": provider_id})
    if df_ev.empty:
        return {"ProviderEpicId": str(provider_id)}  # nothing to classify

    primary = str(df_ev["ApexPrimarySpeciality"].iloc[0] or "").strip()
    if not primary:
        return {"ProviderEpicId": str(provider_id)}  # no specialty → no classification

    # Allowed subs
    allowed_df = pd.read_sql(text(SQL_ALLOWED_SUBS), con=engine, params={"primary_specialty": primary})
    allowed = [row["SubSpeciality"] for _, row in allowed_df.iterrows()]
    if not allowed:
        return {"ProviderEpicId": str(provider_id)}  # no taxonomy defined

    user_prompt = _build_user_prompt(primary, allowed, df_ev.to_dict(orient="records"), str(provider_id))

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        response_format={"type": "json_schema", "json_schema": SUBSPEC_SCHEMA},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    data = json.loads(resp.choices[0].message.content)

    # Post-validate against allowed list and dedupe
    picks = []
    for k in ("DerivedSubSpeciality1", "DerivedSubSpeciality2", "DerivedSubSpeciality3"):
        v = data.get(k)
        if isinstance(v, str) and v.strip() in allowed and v not in picks:
            picks.append(v.strip())

    out = {
        "ProviderEpicId": str(data.get("ProviderEpicId", str(provider_id))),
        "DerivedSubSpeciality1": picks[0] if len(picks) > 0 else None,
        "DerivedSubSpeciality2": picks[1] if len(picks) > 1 else None,
        "DerivedSubSpeciality3": picks[2] if len(picks) > 2 else None,
    }
    return out

def classify_and_update_all(dry_run: bool = False, only_provider_ids: Union[List[str], None] = None):
    # Get provider list (distinct)
    df_prov = pd.read_sql(text(SQL_PROVIDER_LIST), con=engine)
    if only_provider_ids:
        df_prov = df_prov[df_prov["ProviderEpicId"].astype(str).isin([str(x) for x in only_provider_ids])]

    results = []
    for _, row in df_prov.iterrows():
        pid = str(row["ProviderEpicId"]).strip()
        res = _classify_one(pid)
        results.append(res)
        if not dry_run:
            with engine.begin() as conn:  # commit per provider ✅
                exec_result = conn.execute(
                    text(SQL_UPDATE_SUBS),
                    {
                        "s1": res.get("DerivedSubSpeciality1"),
                        "s2": res.get("DerivedSubSpeciality2"),
                        "s3": res.get("DerivedSubSpeciality3"),
                        "provider_id": pid,
                    },
                )
                # Optional progress:
                print(f"Updated {pid} (rows affected: {exec_result.rowcount})")


    # Return a DataFrame of what we applied (handy for logs/QA)
    return pd.DataFrame(results, columns=[
        "ProviderEpicId", "DerivedSubSpeciality1", "DerivedSubSpeciality2", "DerivedSubSpeciality3"
    ])
    
df_all = classify_and_update_all(dry_run=False)
print(df_all.head())
# -------------------------
# Example runs
# -------------------------
# 1) Test just provider 68679 without writing:
# df_preview = classify_and_update_all(dry_run=True, only_provider_ids=["68679"])
# print(df_preview)

# 2) Update 68679 in-place:
# df_applied = classify_and_update_all(dry_run=False, only_provider_ids=["68679"])
# print(df_applied.head())

# 3) Process everyone with data:
# df_all = classify_and_update_all(dry_run=False)
# print(df_all.head())
