#!/usr/bin/env python3
"""
Sparkle vendor terms → UMLS CUI mapping (Linux VM friendly, FULL SciSpaCy linker)

This goes back to the ORIGINAL design:
- Pull Sparkle terms + your provider condition CUIs from SQL (single query / same connection)
- Use scispaCy linker with:
    1) Alias map (fast)
    2) Candidate generator similarity (ANN / nmslib)
    3) NER+linker fallback (kb_ents)
- Produce mapping outputs and unmapped lists

SQL INPUT (expected):
- Sparkle terms: MarketingMirror.NoteExtraction.SparkleHealthMedicalTerms (Type='Condition')
  Columns used: [index], Name
- Provider CUIs / canonical names: MarketingMirror.NoteExtraction.ProviderConditionStats (or temp #sample)
  Columns used: Cui, CanonicalName

OUTPUT CSVs (written next to this script):
- sparkle_to_umls_map.csv
- sparkle_unmapped.csv
- umls_unmapped_by_sparkle.csv

ENV VARS:
- SCUBA_DEV or SCUBA_PRD (SQLAlchemy connection string)
  Example (ODBC):
    export SCUBA_DEV='mssql+pyodbc:///?odbc_connect=DRIVER%3D%7BODBC+Driver+18+for+SQL+Server%7D%3BSERVER%3Dtcp%3Ahost%2C1433%3BDATABASE%3Ddb%3BTrusted_Connection%3DYes%3BEncrypt%3DYes%3BTrustServerCertificate%3DYes'
- SPACY_MODEL (default en_ner_bc5cdr_md)
- SCORE_THRESHOLD (default 0.70)
- TOPK (default 25)
- PREFER_TYPES (default T047,T046,T061,T060)

NOTES:
- On Linux VM, the ANN index load that crashed on Windows should work.
- If you get ODBC errors, install msodbcsql18 + unixodbc in the VM/container.
"""

from __future__ import annotations

import os
import sys
import json
import re
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import spacy
from sqlalchemy import create_engine, text
from scispacy.linking import EntityLinker


# ---------------------------
# Config
# ---------------------------
DEFAULT_PREFER_TYPES = {"T047", "T046", "T061", "T060"}
DEFAULT_SCORE_THRESHOLD = 0.70
DEFAULT_SPACY_MODEL = "en_ner_bc5cdr_md"
DEFAULT_TOPK = 25

# If your CUIs live somewhere else, change this:
DEFAULT_PROVIDER_CUI_SOURCE_SQL = """
SELECT DISTINCT
    s.Cui,
    s.CanonicalName
FROM MarketingMirror.NoteExtraction.ProviderConditionStats s
WHERE s.Cui IS NOT NULL
"""

DEFAULT_SPARKLE_SQL = """
SELECT
    t.[index] AS SparkleIndex,
    t.Name    AS SparkleTerm
FROM MarketingMirror.NoteExtraction.SparkleHealthMedicalTerms t
WHERE t.Type = 'Condition'
  AND t.Name IS NOT NULL
"""


def load_config():
    prefer_raw = os.getenv("PREFER_TYPES", ",".join(sorted(DEFAULT_PREFER_TYPES)))
    prefer_types = {x.strip() for x in prefer_raw.split(",") if x.strip()}

    try:
        score_threshold = float(os.getenv("SCORE_THRESHOLD", str(DEFAULT_SCORE_THRESHOLD)))
    except Exception:
        score_threshold = DEFAULT_SCORE_THRESHOLD

    model_name = os.getenv("SPACY_MODEL", DEFAULT_SPACY_MODEL)

    try:
        topk = int(os.getenv("TOPK", str(DEFAULT_TOPK)))
    except Exception:
        topk = DEFAULT_TOPK

    conn_str = os.getenv("SCUBA_DEV") or os.getenv("SCUBA_PRD")
    if not conn_str:
        raise RuntimeError("Set SCUBA_DEV or SCUBA_PRD to a SQLAlchemy connection string.")

    return prefer_types, score_threshold, model_name, topk, conn_str


# ---------------------------
# Normalization helpers
# ---------------------------
_norm_re = re.compile(r"[^a-z0-9\s\-\/]", re.IGNORECASE)
_space_re = re.compile(r"\s+")


def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = _norm_re.sub(" ", s)
    s = _space_re.sub(" ", s).strip()
    return s


# ---------------------------
# Load spaCy/scispaCy + UMLS linker (FULL)
# ---------------------------
def load_pipeline(model_name: str):
    print(f"Loading spaCy model: {model_name}", flush=True)
    nlp = spacy.load(model_name)
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    linker: EntityLinker = nlp.get_pipe("scispacy_linker")
    kb = linker.kb

    alias_map = getattr(kb, "alias_to_entities", None) or getattr(kb, "alias_to_cuis", None)
    cand_gen = getattr(linker, "candidate_generator", None)
    cand_gen_func = getattr(cand_gen, "generate_candidates", None) if cand_gen else None

    return nlp, kb, alias_map, cand_gen_func


# ---------------------------
# Candidate retrieval helpers
# ---------------------------
def candidates_from_alias(term: str, kb, alias_map) -> List[Tuple[str, float, str, List[str]]]:
    if not alias_map:
        return []
    hits: List[Tuple[str, float, str, List[str]]] = []
    variants = {term, term.lower(), norm_text(term)}
    variants = {v for v in variants if v}

    for t in variants:
        if t not in alias_map:
            continue
        for e in alias_map[t]:
            if isinstance(e, tuple) and len(e) >= 1:
                cui = e[0]
                prior = float(e[1] or 0.0) if len(e) > 1 else 0.0
            else:
                cui = e
                prior = 0.0

            if cui in kb.cui_to_entity:
                ent = kb.cui_to_entity[cui]
                hits.append((cui, prior, ent.canonical_name, list(getattr(ent, "types", []) or [])))

    best: Dict[str, Tuple[float, str, List[str]]] = {}
    for cui, s, name, types in hits:
        if cui not in best or s > best[cui][0]:
            best[cui] = (s, name, types)

    return [(cui, s, name, types) for cui, (s, name, types) in best.items()]


def candidates_from_generator(term: str, kb, cand_gen_func, topk: int) -> List[Tuple[str, float, str, List[str]]]:
    if not cand_gen_func:
        return []
    try:
        cands = cand_gen_func(term)
    except Exception:
        return []

    out: List[Tuple[str, float, str, List[str]]] = []
    # take more than topk; we will rank with type bonus
    for c in cands[: max(topk, topk * 3)]:
        cui = getattr(c, "cui", None)
        score = getattr(c, "score", None)
        if score is None:
            score = getattr(c, "similarity", 0.0)
        if not cui:
            continue
        if cui in kb.cui_to_entity:
            ent = kb.cui_to_entity[cui]
            out.append((cui, float(score or 0.0), ent.canonical_name, list(getattr(ent, "types", []) or [])))
    return out


def candidates_from_ner(term: str, nlp, kb, topk: int) -> List[Tuple[str, float, str, List[str]]]:
    doc = nlp(term)
    out: List[Tuple[str, float, str, List[str]]] = []
    for ent in doc.ents:
        for cui, score in ent._.kb_ents[:topk]:
            if cui in kb.cui_to_entity:
                kbe = kb.cui_to_entity[cui]
                out.append((cui, float(score), kbe.canonical_name, list(getattr(kbe, "types", []) or [])))
    return out


# ---------------------------
# Choose best CUI per term
# ---------------------------
def choose_cui_for_term(
    term: str,
    nlp,
    kb,
    alias_map,
    cand_gen_func,
    prefer_types: Set[str],
    score_threshold: float,
    topk: int,
) -> Optional[dict]:
    def type_bonus(types: List[str]) -> float:
        return 0.1 if any(t in prefer_types for t in (types or [])) else 0.0

    # 1) Alias
    alias_cands = candidates_from_alias(term, kb, alias_map)
    if alias_cands:
        cui, s, name, types = sorted(
            alias_cands,
            key=lambda c: float(c[1]) + type_bonus(c[3]),
            reverse=True,
        )[0]
        return {"cui": cui, "canonical_name": name, "types": list(types or []), "method": "alias", "score": float(s)}

    # 2) Generator
    gen_cands = candidates_from_generator(term, kb, cand_gen_func, topk=topk)
    if gen_cands:
        cui, s, name, types = sorted(
            gen_cands,
            key=lambda c: float(c[1]) + type_bonus(c[3]),
            reverse=True,
        )[0]
        if float(s) + type_bonus(types) >= score_threshold:
            return {"cui": cui, "canonical_name": name, "types": list(types or []), "method": "generator", "score": float(s)}

    # 3) NER fallback
    ner_cands = candidates_from_ner(term, nlp, kb, topk=topk)
    if ner_cands:
        cui, s, name, types = sorted(
            ner_cands,
            key=lambda c: float(c[1]) + type_bonus(c[3]),
            reverse=True,
        )[0]
        if float(s) + type_bonus(types) >= score_threshold:
            return {"cui": cui, "canonical_name": name, "types": list(types or []), "method": "ner", "score": float(s)}

    return None


# ---------------------------
# MAIN
# ---------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    prefer_types, score_threshold, model_name, topk, conn_str = load_config()

    print(f"PREFER_TYPES={prefer_types}", flush=True)
    print(f"SCORE_THRESHOLD={score_threshold}", flush=True)
    print(f"SPACY_MODEL={model_name}", flush=True)
    print(f"TOPK={topk}", flush=True)

    engine = create_engine(conn_str)

    print("Reading Sparkle terms from SQL...", flush=True)
    sparkle = pd.read_sql(DEFAULT_SPARKLE_SQL, engine)
    sparkle["SparkleTerm"] = sparkle["SparkleTerm"].astype(str).str.strip()
    sparkle = sparkle[sparkle["SparkleTerm"] != ""].copy()
    print(f"Sparkle rows: {len(sparkle)}", flush=True)

    print("Reading Provider CUIs from SQL...", flush=True)
    provider_cuis = pd.read_sql(DEFAULT_PROVIDER_CUI_SOURCE_SQL, engine)
    provider_cuis["Cui"] = provider_cuis["Cui"].astype(str).str.strip()
    provider_cuis["CanonicalName"] = provider_cuis["CanonicalName"].astype(str).str.strip()
    provider_cuis = provider_cuis[(provider_cuis["Cui"] != "") & (provider_cuis["CanonicalName"] != "")].drop_duplicates()
    print(f"Provider CUI rows: {len(provider_cuis)}", flush=True)

    # Build a set of allowed CUIs (so we only map to the CUIs your pipeline produced)
    allowed_cuis = set(provider_cuis["Cui"].tolist())

    # Load NLP/linker
    nlp, kb, alias_map, cand_gen_func = load_pipeline(model_name)

    mapped: List[dict] = []
    unmapped: List[dict] = []
    used_cuis: Set[str] = set()

    print("Mapping Sparkle terms → best matching allowed CUI...", flush=True)
    for r in sparkle.itertuples(index=False):
        sparkle_idx = int(r.SparkleIndex)
        sparkle_term = str(r.SparkleTerm).strip()
        if not sparkle_term:
            continue

        chosen = choose_cui_for_term(
            term=sparkle_term,
            nlp=nlp,
            kb=kb,
            alias_map=alias_map,
            cand_gen_func=cand_gen_func,
            prefer_types=prefer_types,
            score_threshold=score_threshold,
            topk=topk,
        )

        if not chosen or chosen["cui"] not in allowed_cuis:
            # If linker returns a CUI not in allowed set, treat as unmapped for this job
            unmapped.append({"SparkleIndex": sparkle_idx, "SparkleTerm": sparkle_term})
            continue

        used_cuis.add(chosen["cui"])
        mapped.append({
            "SparkleIndex": sparkle_idx,
            "SparkleTerm": sparkle_term,
            "Cui": chosen["cui"],
            "UmlsCanonicalName": chosen["canonical_name"],
            "ChosenMethod": chosen["method"],
            "ChosenScore": chosen["score"],
            "Types": json.dumps(chosen["types"]),
        })

    # Provider CUIs that never got matched by any Sparkle term
    umls_unmapped = provider_cuis[~provider_cuis["Cui"].isin(used_cuis)].copy()

    out_map = os.path.join(script_dir, "sparkle_to_umls_map.csv")
    out_unm_sparkle = os.path.join(script_dir, "sparkle_unmapped.csv")
    out_unm_umls = os.path.join(script_dir, "umls_unmapped_by_sparkle.csv")

    pd.DataFrame(mapped).to_csv(out_map, index=False)
    pd.DataFrame(unmapped).to_csv(out_unm_sparkle, index=False)
    umls_unmapped.to_csv(out_unm_umls, index=False)

    print("✅ Done", flush=True)
    print(f"Mapped Sparkle terms: {len(mapped)}", flush=True)
    print(f"Unmapped Sparkle terms: {len(unmapped)}", flush=True)
    print(f"Unmapped Provider CUIs: {len(umls_unmapped)}", flush=True)
    print(f"Wrote: {out_map}", flush=True)
    print(f"Wrote: {out_unm_sparkle}", flush=True)
    print(f"Wrote: {out_unm_umls}", flush=True)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        raise
