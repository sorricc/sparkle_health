from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv() 
conn = os.getenv("WORKSPACE_PRD")
engine = create_engine(conn)

sql = '''
SELECT t.ProviderEpicId,
      t.ClinicalNoteEpicId,
       c.term,
       SUM(c.count) AS count
FROM MarketingWorkSpace.NoteExtraction.NoteConditions AS t
    CROSS APPLY OPENJSON(t.RawConditionsJson, '$.conditions')
                WITH
                (
                    term NVARCHAR(200) '$.term',
                    count INT '$.count'
                ) AS c
    WHERE t.MappedConditions IS NULL AND t.RawConditionsJson IS NOT NULL 
	GROUP by t.ProviderEpicId,ClinicalNoteEpicId,
       c.term
'''

# ✅ Keep ProviderEpicId as string — the same idea as dtype= in read_csv
df = pd.read_sql(sql, engine, dtype={"ProviderEpicId": str})

path = r'C:\Users\sorricc\Documents\GitHub\SparkleHealthNew\UMLS_Mapping\data\Note_terms_input.csv'
df.to_csv(path, index=False)

print(f"✅ Wrote {path}")