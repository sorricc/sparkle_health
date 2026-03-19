import json
import os
from openai import AzureOpenAI
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import re
from tabulate import tabulate
from collections import defaultdict, Counter
import time
import random
from dotenv import load_dotenv


load_dotenv() 

API_KEY = os.getenv("API_KEY_PRD")
API_VERSION = '2024-08-01-preview'
deployment = 'gpt-4o-2024-08-06'
RESOURCE_ENDPOINT = os.getenv("RESOURCE_ENDPOINT_PRD")
conn = os.getenv("SCUBA_PRD")
conn2 = os.getenv("SCUBA_DEV")


client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=RESOURCE_ENDPOINT,
)

engine = create_engine(conn2)

def col_mapping_icd10cm(col):
    if col.name == 'tty':
        return col.map({'PT': 1, 'ET': 2, 'AB': 3, 'FN': 4, 'HT': 5}).fillna(99)
    if col.name == 'ts':
        return np.where(col == 'P', 1, 0)
    if col.name == 'code':
        return col.str.replace('.', '').str.len()
    
def col_mapping_snomed(col):
    if col.name == 'tty':
        return col.map({'PT': 1, 'FN': 2, 'SY': 3}).fillna(99)
    if col.name == 'ts':
        return np.where(col == 'P', 1, 0)
    if col.name == 'str':
        return col.str.len()
    if col.name == 'sab':
        return np.where(col == 'SNOMEDCT_US', 0, 1)

# Pick the best ICD10CM term for each `Cui`
def pick_best_icd10cm(concepts_df, cui):
    # Filter for relevant ICD10CM terms
    filtered = concepts_df[
        (concepts_df["cui"] == cui)
        & (concepts_df["lat"] == "ENG")
        & (concepts_df["sab"] == "ICD10CM")
        & (concepts_df["suppress"] != "Y")
    ]
    # Sort according to the preference rules
    if not filtered.empty:
        filtered = filtered.sort_values(by=['tty',
                                            'ts',
                                            'code'],
                                            ascending=[True, True, False],
                                            key=col_mapping_icd10cm)
    # Return the top result
    return filtered.head(1)

# Pick the best SNOMED term for each `Cui`
def pick_best_snomed(concepts_df, cui):
    # Filter for relevant SNOMED terms
    filtered = concepts_df[
        (concepts_df["cui"] == cui)
        & (concepts_df["lat"] == "ENG")
        & (concepts_df["sab"].isin(["SNOMEDCT_US", "SNOMEDCT"]))
        & (concepts_df["tty"].isin(["PT", "FN", "SY"]))
        & (concepts_df["suppress"] != "Y")
    ]
    if not filtered.empty:
        # Sort according to the preference rules
        filtered = filtered.sort_values(by=['tty',
                                            'sab',
                                            'ts',
                                            'str'],
                                            ascending=[True, True, True, False],
                                            key=col_mapping_snomed)
    # Return the top result
    return filtered.head(1)

# Compute Full TTY Names and ChosenVocab
def compute_tty_name(tty_code, vocab_type):
    if vocab_type == 'ICD10CM':
        return {
            'PT': 'Preferred Term',
            'ET': 'Entry Term',
            'AB': 'Abbreviation',
            'FN': 'Fully Specified Name',
            'HT': 'Hierarchical Term'
        }.get(tty_code, None)
    elif vocab_type == 'SNOMEDCT':
        return {
            'PT': 'Preferred Term',
            'FN': 'Fully Specified Name',
            'SY': 'Synonym'
        }.get(tty_code, None)
    return None

sql_query1 = text("""
SELECT * FROM NoteExtraction.ProviderConditionsFinal
""")

provider_conditions_final_df = pd.read_sql(sql_query1, engine)
processed_data = []


# Iterate through each row in provider_conditions_final
for index, row in provider_conditions_final_df.iterrows():
    # Parse the JSON data in the `MappedConditions` column
    mapped_conditions = json.loads(row["MappedConditions"])
    
    # Extract relevant fields and filter rows where `total_count > 1`
    for condition in mapped_conditions:
        if int(condition["total_count"]) > 1:
            processed_data.append({
                "ProviderEpicId": row["ProviderEpicId"],
                "Cui": condition["cui"],
                "CanonicalName": condition["canonical_name"],
                "TotalCount": int(condition["total_count"])
            })

# Create the temporary table (DataFrame) from the processed data
standard_terms_df = pd.DataFrame(processed_data)

# Display the final DataFrame
print("Temporary Table (#StandardTerms):")
print(standard_terms_df)

cui_list = standard_terms_df['Cui'].tolist()
cui_values = ",".join(f"'{cui}'" for cui in cui_list)

# Initialize coverage cutoff
coverage_cutoff = 0.80
specialty = 'All'

# Base
base = standard_terms_df[standard_terms_df['TotalCount'] > 0][['ProviderEpicId', 'Cui', 'CanonicalName', 'TotalCount']]

# WithTotals
with_totals = base.copy()
with_totals['ProviderTotal'] = with_totals.groupby('ProviderEpicId')['TotalCount'].transform('sum')

# Ranked
ranked = with_totals.copy()
ranked['ConditionShare'] = ranked['TotalCount'] / ranked['ProviderTotal'].replace(0, np.nan)
ranked['DenseRankDesc'] = ranked.groupby('ProviderEpicId')['TotalCount'].rank(method='dense', ascending=False)

# Grouped
grouped = ranked.copy()
grouped['TieGroupCount'] = grouped.groupby(['ProviderEpicId', 'DenseRankDesc'])['TotalCount'].transform('sum')

# Coverage
coverage = grouped.copy()
coverage['CumulativeShare_TieAware'] = grouped.groupby('ProviderEpicId')['TieGroupCount'].cumsum() / grouped['ProviderTotal'].replace(0, np.nan)

# Distribution
distribution = coverage.copy()
distribution['Decile'] = distribution.sort_values(by=['ProviderEpicId', 'TotalCount', 'Cui'], ascending=[True, False, True]).groupby('ProviderEpicId').cumcount().add(1).div(distribution.groupby('ProviderEpicId')['Cui'].transform('count')).mul(10).apply(np.ceil).astype(int)
distribution['Quartile'] = distribution.sort_values(by=['ProviderEpicId', 'TotalCount', 'Cui'], ascending=[True, False, True]).groupby('ProviderEpicId').cumcount().floordiv(distribution.groupby('ProviderEpicId')['Cui'].transform('count') // 4).add(1)

distribution['CumeDistDescOrder'] = distribution.sort_values(by=['ProviderEpicId', 'TotalCount', 'Cui'], ascending=[True, False, True]).groupby('ProviderEpicId').cumcount().add(1).div(distribution.groupby('ProviderEpicId')['Cui'].transform('count'))
# distribution['Decile'] = distribution.groupby('ProviderEpicId')['TotalCount'].rank(method='first', ascending=False, pct=True) * 10
# distribution['Quartile'] = distribution.groupby('ProviderEpicId')['TotalCount'].rank(method='first', ascending=False, pct=True) * 4
# distribution['CumeDistDescOrder'] = distribution.groupby('ProviderEpicId')['TotalCount'].rank(method='max', ascending=False) / distribution.groupby('ProviderEpicId')['TotalCount'].transform('count')

# WithZ
with_z = distribution.copy()
with_z['CntMean'] = with_z.groupby('ProviderEpicId')['TotalCount'].transform('mean')
with_z['CntStdDev'] = with_z.groupby('ProviderEpicId')['TotalCount'].transform('std')

# ProviderStats
total_providers = base['ProviderEpicId'].nunique()

# ConditionStats
condition_stats = base.groupby('Cui')['ProviderEpicId'].nunique().reset_index()
condition_stats.columns = ['Cui', 'ProvidersTreatingThisCui']

provider_list = with_z['ProviderEpicId'].tolist()
provider_values = ",".join(f"'{p}'" for p in provider_list)

# get current providers
sql_query3 = f"""
SELECT * FROM HealthDW.baseview.ProviderDim AS pd
WHERE pd.IsCurrent = 1 AND pd.EpicProviderId IN ({provider_values})
"""

provider_dim_df = pd.read_sql(sql_query3, engine)

# Final calculations
final = with_z.copy()
final = final.merge(condition_stats, on='Cui', how='left')
# final = final.merge(term_vocab, on=['ProviderEpicId', 'Cui'], how='left')
final = final.merge(provider_dim_df[provider_dim_df['IsCurrent'] == 1], left_on='ProviderEpicId', right_on='EpicProviderId', how='inner')

# Add derived columns
final['TierLabel'] = np.where(final['Decile'] == 1, 'Top Decile',
                              np.where(final['Quartile'] == 1, 'Top Quartile',
                                       np.where(final['CumeDistDescOrder'] <= 0.50, 'Top Half', 'Lower Half')))
final['CoverageLabel'] = np.where(final['CumulativeShare_TieAware'] <= 0.5, 'Core',
                                  np.where(final['CumulativeShare_TieAware'] <= 0.8, 'Major', 'Peripheral'))
final['ZScoreWithinProvider'] = np.where(final['CntStdDev'].isnull() | (final['CntStdDev'] == 0), None,
                                         (final['TotalCount'] - final['CntMean']) / final['CntStdDev'])
final['TFIDF_SpecialtyScore'] = final['TotalCount'] * np.log(total_providers / final['ProvidersTreatingThisCui'].replace(0, np.nan))

# Filter and sort final output
specialty_filter = (specialty == 'All') | (final['PrimarySpecialty'] == specialty)
final_filtered = final[specialty_filter]
final_sorted = final_filtered.sort_values(by=['ProviderEpicId', 'Decile', 'TotalCount', 'Cui'], ascending=[True, True, False, True])

# Join with ProviderConditionsFinal and produce output
output = final_sorted.merge(provider_conditions_final_df, on='ProviderEpicId', how='inner')
output = output[['ProviderEpicId', 'ProviderName', 'PrimarySpecialty', 'DerivedSubSpeciality1', 'DerivedSubSpeciality2',
                 'DerivedSubSpeciality3', 'Cui', 'CanonicalName', 'TotalCount', 'ProviderTotal', 'ConditionShare',
                 'CumulativeShare_TieAware', 'Decile', 'Quartile', 'TierLabel', 'CoverageLabel', 'ZScoreWithinProvider', 'TFIDF_SpecialtyScore']]
                #  'ICD10_Code', 'ICD10CM_Description', 'ICD10CM_TTY', 'SNOMED_Description',
                #  'SNOMED_TTY']]

print(output)