#!/usr/bin/env python3
"""
Note-level UMLS mapping (standalone) — SQL-friendly JSON outputs

INPUT CSV schema (required):
- ClinicalNoteEpicId (str)
- ProviderEpicId     (str)
- term               (str)     e.g., "type 2 diabetes mellitus"
- count              (int)     frequency for that note/term

OUTPUT CSV schema:
- ClinicalNoteEpicId (str)
- ProviderEpicId     (str)
- MappedByCui        (nvarchar(max) JSON)  -- {"CUI": {canonical_name, aliases[], chosen_method, chosen_score, ...}, ...}
- UnmappedConditions (nvarchar(max) JSON)  -- [{"term":"...", "count":N}, ...]

BEHAVIOR:
- For each (ClinicalNoteEpicId, ProviderEpicId) group:
  - Map each term to a UMLS CUI via:
      1) Alias map (fast)
      2) Candidate generator (nmslib similarity)
      3) NER+linker (fallback)
  - Collapse synonyms to the same CUI (sum counts, union aliases).
  - Keep unmapped terms in a separate array (UnmappedConditions).
- Record method & score details so you can threshold/filter in SQL later.

ENV TUNING (optional):
- PREFER_TYPES="T047,T046"     # Bias to Disease/Syndrome, Pathologic Function
- SCORE_THRESHOLD="0.1"        # 0..1; recommended 0.1 for clinical terms from LLM
- SPACY_MODEL="en_ner_bc5cdr_md"

REQUIRES:
- pandas>=2.0,<3.0
- spacy>=3.7,<4.0
- scispacy>=0.5.3,<0.6
- (and the spaCy model specified by SPACY_MODEL)
- Entity Linker Model: 
- Use following command to install: 
- pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
- Python < 3.9
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd
import spacy
from scispacy.linking import EntityLinker

# ---------------------------
# Args & config
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Note-level UMLS mapping with SQL-friendly JSON columns")
    ap.add_argument("--in",  dest="in_csv",  required=True, help="Input CSV path")
    ap.add_argument("--out", dest="out_csv", required=True, help="Output CSV path")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    return ap.parse_args()

def log(msg: str, verbose: bool):
    if verbose:
        print(msg, flush=True)

def load_env():
    prefer_raw = os.getenv("PREFER_TYPES", "T047,T046")
    prefer_types = {x.strip() for x in prefer_raw.split(",") if x.strip()}
    try:
        score_threshold = float(os.getenv("SCORE_THRESHOLD", "0.1"))  # default 0.1 (precision-oriented)
    except Exception:
        score_threshold = 0.1
    model_name = os.getenv("SPACY_MODEL", "en_ner_bc5cdr_md")
    return prefer_types, score_threshold, model_name

# ---------------------------
# Load spaCy/scispaCy + UMLS linker
# ---------------------------
def load_pipeline(model_name: str, verbose: bool):
    log(f"Loading spaCy model: {model_name}", verbose)
    nlp = spacy.load(model_name)
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    linker: EntityLinker = nlp.get_pipe("scispacy_linker")
    kb = linker.kb

    alias_map = getattr(kb, "alias_to_entities", None) or getattr(kb, "alias_to_cuis", None)
    cand_gen = getattr(linker, "candidate_generator", None)
    cand_gen_func = getattr(cand_gen, "generate_candidates", None)

    meta = {
        "model_name": model_name,
        "linker_name": "umls",
        "kb_version": getattr(kb, "version", None),
    }
    return nlp, linker, kb, alias_map, cand_gen_func, meta

# ---------------------------
# Candidate retrieval helpers
# ---------------------------
def candidates_from_alias(term: str, kb, alias_map):
    if not alias_map:
        return []
    hits = []
    for t in {term, term.lower()}:
        if t in alias_map:
            entries = alias_map[t]
            for e in entries:
                if isinstance(e, tuple) and len(e) >= 2:
                    cui, prior = e[0], float(e[1] or 0.0)
                else:
                    cui, prior = (e, 0.0)
                if cui in kb.cui_to_entity:
                    ent = kb.cui_to_entity[cui]
                    hits.append((cui, prior, ent.canonical_name, getattr(ent, "types", [])))
    # Deduplicate by cui, keep highest prior
    best = {}
    for cui, s, name, types in hits:
        if cui not in best or s > best[cui][0]:
            best[cui] = (s, name, types)
    return [(cui, s, name, types) for cui, (s, name, types) in best.items()]

def candidates_from_generator(term: str, kb, cand_gen_func):
    if not cand_gen_func:
        return []
    try:
        cands = cand_gen_func(term)
    except Exception:
        return []
    out = []
    for c in cands:
        cui = getattr(c, "cui", None)
        score = getattr(c, "score", None) or getattr(c, "similarity", 0.0)
        if cui in kb.cui_to_entity:
            ent = kb.cui_to_entity[cui]
            out.append((cui, float(score or 0.0), ent.canonical_name, getattr(ent, "types", [])))
    return out

def candidates_from_ner(term: str, nlp, kb):
    doc = nlp(term)
    out = []
    for ent in doc.ents:
        for cui, score in ent._.kb_ents:
            if cui in kb.cui_to_entity:
                kbe = kb.cui_to_entity[cui]
                out.append((cui, float(score), kbe.canonical_name, getattr(kbe, "types", [])))
    return out

# ---------------------------
# Choose best CUI per term (returns method & scores for auditing)
# ---------------------------
def choose_cui_for_term(term: str,
                        nlp,
                        kb,
                        alias_map,
                        cand_gen_func,
                        prefer_types: Set[str],
                        score_threshold: float):
    """
    Returns:
      dict or None, with keys:
        - cui, canonical_name, types (list[str])
        - chosen_method: 'alias' | 'generator' | 'ner'
        - alias_prior, generator_similarity, ner_confidence (floats or None)
        - type_bonus_applied (float), prefer_type_hit (bool)
        - chosen_score (float)  # score actually used for ranking
    """
    # 1) Alias path
    alias_cands = candidates_from_alias(term, kb, alias_map)
    if alias_cands:
        if len(alias_cands) == 1:
            cui, s, name, types = alias_cands[0]
            type_hit = any(t in prefer_types for t in (types or []))
            type_bonus = 0.1 if type_hit else 0.0
            chosen_score = s + type_bonus
            # alias path ignores threshold (treat as exact-ish)
            return {
                "cui": cui, "canonical_name": name, "types": list(types or []),
                "chosen_method": "alias",
                "alias_prior": float(s), "generator_similarity": None, "ner_confidence": None,
                "type_bonus_applied": type_bonus, "prefer_type_hit": bool(type_hit),
                "chosen_score": float(chosen_score),
            }
        # Rank multiple alias candidates (optionally blend generator scores)
        gen_scores = {cui: s for cui, s, *_ in candidates_from_generator(term, kb, cand_gen_func)} if cand_gen_func else {}
        def rank_alias(c):
            cui, s, name, types = c
            type_bonus = 0.1 if any(t in prefer_types for t in (types or [])) else 0.0
            # Prefer generator corroboration if available; otherwise alias prior
            base = gen_scores.get(cui, s if not gen_scores else 0.0)
            return base + type_bonus
        best = sorted(alias_cands, key=rank_alias, reverse=True)[0]
        cui, s, name, types = best
        type_hit = any(t in prefer_types for t in (types or []))
        type_bonus = 0.1 if type_hit else 0.0
        chosen_score = (gen_scores.get(cui, 0.0) or 0.0) + (type_bonus if gen_scores else 0.0)
        if not gen_scores:
            chosen_score = s + type_bonus
        return {
            "cui": cui, "canonical_name": name, "types": list(types or []),
            "chosen_method": "alias",
            "alias_prior": float(s), "generator_similarity": gen_scores.get(cui, None), "ner_confidence": None,
            "type_bonus_applied": type_bonus, "prefer_type_hit": bool(type_hit),
            "chosen_score": float(chosen_score),
        }

    # 2) Generator path
    gen_list = candidates_from_generator(term, kb, cand_gen_func)
    if gen_list:
        def rank_gen(c):
            cui, s, name, types = c
            return float(s) + (0.1 if any(t in prefer_types for t in (types or [])) else 0.0)
        top = sorted(gen_list, key=rank_gen, reverse=True)[0]
        cui, s, name, types = top
        type_hit = any(t in prefer_types for t in (types or []))
        type_bonus = 0.1 if type_hit else 0.0
        chosen_score = float(s) + type_bonus
        if chosen_score >= score_threshold:
            return {
                "cui": cui, "canonical_name": name, "types": list(types or []),
                "chosen_method": "generator",
                "alias_prior": None, "generator_similarity": float(s), "ner_confidence": None,
                "type_bonus_applied": type_bonus, "prefer_type_hit": bool(type_hit),
                "chosen_score": float(chosen_score),
            }

    # 3) NER path
    ner_list = candidates_from_ner(term, nlp, kb)
    if ner_list:
        top = sorted(ner_list, key=lambda c: c[1], reverse=True)[0]
        cui, s, name, types = top
        type_hit = any(t in prefer_types for t in (types or []))
        type_bonus = 0.1 if type_hit else 0.0
        chosen_score = float(s) + type_bonus
        if chosen_score >= score_threshold:
            return {
                "cui": cui, "canonical_name": name, "types": list(types or []),
                "chosen_method": "ner",
                "alias_prior": None, "generator_similarity": None, "ner_confidence": float(s),
                "type_bonus_applied": type_bonus, "prefer_type_hit": bool(type_hit),
                "chosen_score": float(chosen_score),
            }

    return None

# Global cache to speed repeated terms across notes
TERM_CACHE: Dict[str, Optional[dict]] = {}

# ---------------------------
# Map one note → (MappedByCui dict, UnmappedConditions list)
# ---------------------------
def map_terms_for_note(note_df: pd.DataFrame,
                       nlp,
                       kb,
                       alias_map,
                       cand_gen_func,
                       prefer_types: Set[str],
                       score_threshold: float):
    """
    Returns:
      mapped_by_cui: dict keyed by CUI
      unmapped_conditions: list[{"term": str, "count": int}]
    """
    by_cui = defaultdict(lambda: {
        "canonical_name": None,
        "aliases": set(),
        "types": None,
        "chosen_method": None,
        "chosen_score": None,
        "alias_prior": None,
        "generator_similarity": None,
        "ner_confidence": None,
        "type_bonus_applied": 0.0,
        "prefer_type_hit": False,
    })
    unmapped_counts = defaultdict(int)

    for _, row in note_df.iterrows():
        term = (row["term"] or "").strip()
        cnt  = int(row["count"] or 0)
        if not term or cnt <= 0:
            continue

        if term not in TERM_CACHE:
            TERM_CACHE[term] = choose_cui_for_term(
                term, nlp, kb, alias_map, cand_gen_func, prefer_types, score_threshold
            )
        chosen = TERM_CACHE[term]

        if not chosen:
            unmapped_counts[term] += cnt
            continue

        cui = chosen["cui"]
        slot = by_cui[cui]
        slot["canonical_name"] = slot["canonical_name"] or chosen["canonical_name"]
        slot["types"] = slot["types"] or list(chosen.get("types") or [])
        # Retain the highest scoring method if same CUI appears via different terms
        prev_score = slot["chosen_score"] if slot["chosen_score"] is not None else -1.0
        if chosen["chosen_score"] is not None and chosen["chosen_score"] > prev_score:
            slot["chosen_method"] = chosen["chosen_method"]
            slot["chosen_score"] = chosen["chosen_score"]
            slot["alias_prior"] = chosen["alias_prior"]
            slot["generator_similarity"] = chosen["generator_similarity"]
            slot["ner_confidence"] = chosen["ner_confidence"]
            slot["type_bonus_applied"] = chosen["type_bonus_applied"]
            slot["prefer_type_hit"] = chosen["prefer_type_hit"]
        slot["aliases"].add(term)

    # Finalize shapes
    mapped_by_cui: Dict[str, dict] = {}
    for cui, data in by_cui.items():
        mapped_by_cui[cui] = {
            "canonical_name": data["canonical_name"],
            "aliases": sorted(list(data["aliases"])),
            "types": data["types"],
            "chosen_method": data["chosen_method"],
            "chosen_score": data["chosen_score"],
            "alias_prior": data["alias_prior"],
            "generator_similarity": data["generator_similarity"],
            "ner_confidence": data["ner_confidence"],
            "type_bonus_applied": data["type_bonus_applied"],
            "prefer_type_hit": data["prefer_type_hit"],
        }

    unmapped_conditions = [
        {"term": t, "count": int(c)} for t, c in sorted(unmapped_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    return mapped_by_cui, unmapped_conditions

# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    verbose = args.verbose

    prefer_types, score_threshold, model_name = load_env()
    if verbose:
        print(f"PREFER_TYPES={prefer_types}  SCORE_THRESHOLD={score_threshold}  SPACY_MODEL={model_name}", flush=True)

    # Load input CSV
    log(f"Reading {args.in_csv}", verbose)
    df = pd.read_csv(args.in_csv, dtype={"ClinicalNoteEpicId": str, "ProviderEpicId": str})
    # Normalize
    df["term"] = df["term"].astype(str).str.strip()
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df = df[(df["term"] != "") & (df["count"] > 0)]

    required_cols = {"ClinicalNoteEpicId", "ProviderEpicId", "term", "count"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    # Load NLP/linker
    nlp, linker, kb, alias_map, cand_gen_func, meta = load_pipeline(model_name, verbose)
    log(f"Linker meta: {meta}", verbose)

    # Group per (note, provider)
    note_records: List[dict] = []
    grouped = df.groupby(["ClinicalNoteEpicId", "ProviderEpicId"], dropna=False)
    total_groups = len(grouped)
    log(f"Processing {total_groups} notes...", verbose)

    for (note_id, provider_id), g in grouped:
        mapped_by_cui, unmapped_conditions = map_terms_for_note(
            g, nlp, kb, alias_map, cand_gen_func, prefer_types, score_threshold
        )
        note_records.append({
            "ClinicalNoteEpicId": str(note_id),
            "ProviderEpicId": str(provider_id),
            "MappedByCui": json.dumps(mapped_by_cui, ensure_ascii=False),
            "UnmappedConditions": json.dumps(unmapped_conditions, ensure_ascii=False),
        })

    out = pd.DataFrame(
        note_records,
        columns=["ClinicalNoteEpicId", "ProviderEpicId", "MappedByCui", "UnmappedConditions"]
    )
    out["ClinicalNoteEpicId"] = out["ClinicalNoteEpicId"].astype(str)
    out["ProviderEpicId"]     = out["ProviderEpicId"].astype(str)

    out.to_csv(args.out_csv, index=False)
    print(f"✅ Saved {args.out_csv}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


