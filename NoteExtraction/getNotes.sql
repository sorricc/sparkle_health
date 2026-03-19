DROP TABLE IF EXISTS #provs;
SELECT DISTINCT
       m.NPI,
       pd.EpicProviderId AS ProviderEpicId,
	   pd.PrimarySpecialty
INTO #provs
FROM EchoMirror.dbo.Marketing m
    INNER JOIN HealthDW.base.ProviderDim pd
        ON m.NPI = pd.ProviderNPI
           AND pd.ProviderNPI <> ''
WHERE pd.ProviderPersonType = 'EMPLOYEE'
      AND pd.HrEmployeeStatus = 'Active'
      AND pd.IsCurrent = 1
      AND m.NPI <> ''
	  AND PrimarySpecialty LIKE '%Ortho%'
	  AND pd.IsPhysician = 1
	  AND pd.HasSchedulingTemplate = 1

DECLARE @StartDateKey INT = CONVERT(INT, FORMAT(DATEADD(YEAR, -1, GETDATE()), 'yyyyMMdd'));
DECLARE @EndDateKey INT = CONVERT(INT, FORMAT(GETDATE(), 'yyyyMMdd'));
DROP TABLE IF EXISTS #notes
SELECT 
       p.ProviderEpicId,
       f.ClinicalNoteKey,
       f.ClinicalNoteEpicId,
       ef.EncounterEpicCsn,
       f.Type,
       f.Service,
       f.ServiceInstant,
	   f.Status,
	   pp.PrimarySpecialty 
INTO #notes
FROM CDW.FilteredAccess.ClinicalNoteFact f
    INNER JOIN CDW.FilteredAccess.ProviderDim p
        ON f.AuthoringProviderDurableKey = p.DurableKey
    INNER JOIN #provs pp
        ON p.ProviderEpicId = pp.ProviderEpicId
    INNER JOIN CDW.FilteredAccess.EncounterFact ef
        ON f.EncounterKey = ef.EncounterKey
WHERE p.IsCurrent = 1
      AND f.ServiceDateKey  BETWEEN @StartDateKey AND @EndDateKey
      AND p.DurableKey > 0
      AND f.Type IN ( 'H&P', 'Progress Notes' )
	  AND f.Status = 'Signed'
	  AND NOT EXISTS (SELECT 1 FROM MarketingWorkSpace.NoteExtraction.NoteConditions nc WHERE nc.ClinicalNoteEpicId = f.ClinicalNoteEpicId)
--ORDER BY ServiceInstant DESC;




INSERT INTO MarketingWorkSpace.NoteExtraction.NoteConditions
(
ClinicalNoteTextKey,
    ClinicalNoteEpicId,
    ProviderEpicId,
    EncounterEpicCsn,
    Type,
    PrimarySpecialty

)

SELECT txt.ClinicalNoteTextKey,
       p.ClinicalNoteEpicId,
       p.ProviderEpicId,
       p.EncounterEpicCsn,
       p.Type,
       PrimarySpecialty
FROM #notes p
    INNER JOIN CDW.FilteredAccess.ClinicalNoteTextFact txt
        ON p.ClinicalNoteKey = txt.ClinicalNoteKey

WHERE NOT EXISTS (SELECT 1 FROM MarketingWorkSpace.NoteExtraction.NoteConditions nc WHERE nc.ClinicalNoteEpicId = p.ClinicalNoteEpicId)