from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os
import json
import ast
from sqlalchemy.dialects.mssql import NVARCHAR, INTEGER, TIMESTAMP

"""
This script was used to create ONE-TIME ConditionConsumerTermsV2. The V2 version is a MCP integration friendly format of ConditionConsumerTerms.
There is no difference in data, only the column format and number of column is different.
We don't need to run this file again as the main python script getConsumerTerms.py is updated to output the data in new format.
"""

load_dotenv() 
conn = os.getenv("SCUBA_DEV")
engine = create_engine(conn)

sql = '''
SELECT *
  FROM [MarketingMirror].[NoteExtraction].[ConditionConsumerTerms]
'''

df = pd.read_sql(sql, engine)
df['TermsJson'] = df['TermsJson'].to_dict()
df['TermsJson'] = df['TermsJson'].apply(lambda x:ast.literal_eval(x))
df['preferred_name'] = df['TermsJson'].apply(lambda x:x['preferred_name'])
df['idx_cui(cui)'] = df['TermsJson'].apply(lambda x:x['cui'])
df['terms'] = df['TermsJson'].apply(lambda x:x['terms'])
df['TermsJson'] = df['TermsJson'].apply(lambda x:json.dumps(x))
df['terms'] = df['terms'].apply(lambda x:json.dumps(x))
df.rename(columns={'TermsJson':'idx_terms_fulltext(preferred_name)',
                   'Id':'id',
                   'CUI':'cui',
                   'Model': 'model',
                   'PromptHash': 'prompt_hash',
                   'CreatedAt': 'created_at'},
                   inplace=True)

with engine.begin() as conn:
    # If you want pandas to create columns with specific types on first run:
    dtype_map = {
        "id": INTEGER,
        "cui": NVARCHAR(None),
        "created_at": TIMESTAMP,
        "preferred_name": NVARCHAR(None),
        "idx_cui(cui)": NVARCHAR(None),
    }
df.to_sql(
        "ConditionConsumerTermsV2",
        con=engine,
        schema='NoteExtraction',
        if_exists="append",
        index=False
    )


print(f"Done. Inserted {len(df)} row(s) into NoteExtraction.ConditionConsumerTermsV2.")