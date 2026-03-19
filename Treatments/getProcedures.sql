DROP TABLE IF EXISTS #temp;
DECLARE @StartDateKey INT = CONVERT(INT, FORMAT(DATEADD(YEAR, -2, GETDATE()), 'yyyyMMdd'));
DECLARE @EndDateKey INT = CONVERT(INT, FORMAT(GETDATE(), 'yyyyMMdd'));
WITH ExcludeEncs
AS (SELECT s.EncounterKey
    FROM MarketingWorkSpace.NoteExtraction.ProviderTreatmentDetails s),
     Provs
AS (SELECT ProviderEpicId
    FROM MarketingWorkSpace.NoteExtraction.NoteConditions pcf
    GROUP BY ProviderEpicId)
SELECT DISTINCT
       CONCAT(
                 COALESCE(HospitalBillingTransactionEpicId, ProfessionalBillingTransactionEpicId),
                 '|',
                 pd.BillingProcedureEpicId
             ) AS NaturalId,
       NaturalField = 'Transaction Id | PROC_ID',
       SourceKey = 'Apex',
       baf.AccountEpicId,
       ef.EncounterEpicCsn,
       IIF(dep.DepartmentEpicId = '*Not Applicable', NULL, dep.DepartmentEpicId) AS ApexDepartmentId,
       dep.LocationEpicId,
       EventType = CONCAT(
                             'Billed ',
                             CASE
                                 WHEN RIGHT(Outpatient_Procedure_Group_Category, 1) = 's'
                                      AND Outpatient_Procedure_Group_Category <> 'DME and Supplies' THEN
                                     LEFT(Outpatient_Procedure_Group_Category, LEN(Outpatient_Procedure_Group_Category)
                                                                               - 1)
                                 ELSE
                                     COALESCE(Outpatient_Procedure_Group_Category, pd.Category)
                             END
                         ),
       EventDetail = COALESCE(CPT_HCPCS_Revenue_Code_Description, pd.Name),
       EventSource = 'Apex Transaction',
       -- pef.EncounterKey,
       pef.BillingProcedureCode AS BillingCode,
       CASE
           WHEN pd.CptCode <> '' THEN
               'CPT Code'
           WHEN pd.HcpcsCode <> '' THEN
               'HCPCS Code'
           ELSE
               'Custom'
       END AS CodeType,
       c.Outpatient_Procedure_Group AS ProcedureGroup,
       [Procedure] = CASE
                         WHEN CHARINDEX(':', Outpatient_Procedure) > 0 THEN
                             LTRIM(SUBSTRING(
                                                Outpatient_Procedure,
                                                CHARINDEX(':', Outpatient_Procedure) + 1,
                                                LEN(Outpatient_Procedure) - CHARINDEX(':', Outpatient_Procedure)
                                            )
                                  )
                         ELSE
                             Outpatient_Procedure
                     END,
       SUM(pef.ChargeAmount) AS Amount,
       AmountType = 'Charge',
       pef.BillingSystemType,
       pef.ServiceDateKey,
       pef.PatientDurableKey,
       prov.ProviderEpicId,
       pef.EncounterKey,
       pd.Name,
       pd.PatientFriendlyName,
       pd.Category
INTO #temp
FROM CDW.FilteredAccess.BillingTransactionFact pef
    INNER JOIN CDW.FilteredAccess.ProviderDim prov
        ON pef.BillingProviderDurableKey = prov.DurableKey
           AND prov.IsCurrent = 1
    INNER JOIN Provs
        ON prov.ProviderEpicId = Provs.ProviderEpicId
    INNER JOIN CDW.FilteredAccess.BillingAccountFact baf
        ON pef.BillingAccountKey = baf.BillingAccountKey
    INNER JOIN CDW.FilteredAccess.EncounterFact ef
        ON pef.EncounterKey = ef.EncounterKey
           AND pef.EncounterKey > 0
    LEFT OUTER JOIN ExcludeEncs e
        ON pef.EncounterKey = e.EncounterKey
    INNER JOIN CDW.FilteredAccess.DepartmentDim dep
        ON pef.DepartmentKey = dep.DepartmentKey
    INNER JOIN CDW.FilteredAccess.BillingProcedureDim pd
        ON pef.BillingProcedureDurableKey = pd.DurableKey
           AND pd.IsCurrent = 1
    INNER JOIN MarketingCloudMirror.dbo.CptCodeSet c
        ON c.CPT_HCPCS_Revenue_Code = pef.BillingProcedureCode
WHERE pef.TransactionType = 'Charge'
      AND e.EncounterKey IS NULL
      AND pef.ServiceDateKey
      BETWEEN @StartDateKey AND @EndDateKey
      AND c.Outpatient_Procedure_Group IN ( 'Procedures - Major', 'Procedures - Minor', 'Endoscopy',
                                            'Radiation Therapy', 'Rehab', 'Behavioral Health Services', 'Dental',
                                            'Advanced Imaging - MRI', 'Advanced Imaging - CT',
                                            'Advanced Imaging - PET', 'Standard Imaging - US',
                                            'Standard Imaging - X-Ray', 'Standard Imaging - Nuclear Med/SPECT',
                                            'Diagnostics'
                                          )
GROUP BY CONCAT(
                   COALESCE(HospitalBillingTransactionEpicId, ProfessionalBillingTransactionEpicId),
                   '|',
                   pd.BillingProcedureEpicId
               ),
         baf.AccountEpicId,
         ef.EncounterEpicCsn,
         IIF(dep.DepartmentEpicId = '*Not Applicable', NULL, dep.DepartmentEpicId),
         dep.LocationEpicId,
         CONCAT(
                   'Billed ',
                   CASE
                       WHEN RIGHT(Outpatient_Procedure_Group_Category, 1) = 's'
                            AND Outpatient_Procedure_Group_Category <> 'DME and Supplies' THEN
                           LEFT(Outpatient_Procedure_Group_Category, LEN(Outpatient_Procedure_Group_Category) - 1)
                       ELSE
                           COALESCE(Outpatient_Procedure_Group_Category, pd.Category)
                   END
               ),
         COALESCE(CPT_HCPCS_Revenue_Code_Description, pd.Name),
         pef.BillingProcedureCode,
         CASE
             WHEN pd.CptCode <> '' THEN
                 'CPT Code'
             WHEN pd.HcpcsCode <> '' THEN
                 'HCPCS Code'
             ELSE
                 'Custom'
         END,
         c.Outpatient_Procedure_Group,
         CASE
             WHEN CHARINDEX(':', Outpatient_Procedure) > 0 THEN
                 LTRIM(SUBSTRING(
                                    Outpatient_Procedure,
                                    CHARINDEX(':', Outpatient_Procedure) + 1,
                                    LEN(Outpatient_Procedure) - CHARINDEX(':', Outpatient_Procedure)
                                )
                      )
             ELSE
                 Outpatient_Procedure
         END,
         pef.BillingSystemType,
         pef.ServiceDateKey,
         pef.PatientDurableKey,
         prov.ProviderEpicId,
         pef.EncounterKey,
         pd.Name,
         pd.PatientFriendlyName,
         pd.Category
HAVING SUM(pef.ChargeAmount) > 0;



DROP TABLE IF EXISTS #base;
SELECT BillingCode = UPPER(LTRIM(RTRIM(COALESCE(NULLIF(bpd.CptCode, ''), NULLIF(bpd.HcpcsCode, ''))))),
       CodeType = CASE
                      WHEN UPPER(LTRIM(RTRIM(COALESCE(NULLIF(bpd.CptCode, ''), NULLIF(bpd.HcpcsCode, ''))))) LIKE '[A-Z]%' THEN
                          'HCPCS'
                      WHEN UPPER(LTRIM(RTRIM(COALESCE(NULLIF(bpd.CptCode, ''), NULLIF(bpd.HcpcsCode, ''))))) LIKE '[0-9][0-9][0-9][0-9][0-9]' THEN
                          'CPT'
                      ELSE
                          NULL
                  END,
       CodeDescription = NULLIF(LTRIM(RTRIM(bpd.Name)), ''),
       PatientFriendlyName = NULLIF(LTRIM(RTRIM(bpd.PatientFriendlyName)), '')
INTO #base
FROM CDW.FilteredAccess.BillingProcedureDim bpd
WHERE COALESCE(NULLIF(LTRIM(RTRIM(bpd.CptCode)), ''), NULLIF(LTRIM(RTRIM(bpd.HcpcsCode)), '')) IS NOT NULL
      AND bpd.IsCurrent = 1;


DROP TABLE IF EXISTS #codes;
SELECT b.BillingCode,
       CodeType = MAX(b.CodeType),
       DescriptionsJson =
       (
           SELECT DISTINCT
                  d.CodeDescription,
                  d.PatientFriendlyName
           FROM #base d
           WHERE d.BillingCode = b.BillingCode
           FOR JSON PATH
       )
INTO #codes
FROM #base b
GROUP BY b.BillingCode;


DELETE t
FROM #temp t
    INNER JOIN MarketingWorkSpace.NoteExtraction.ProviderTreatmentDetails s
        ON t.NaturalId = s.NaturalId;


INSERT INTO MarketingWorkSpace.NoteExtraction.ProviderTreatmentDetails
(
    NaturalId,
    NaturalField,
    SourceKey,
    AccountEpicId,
    EncounterEpicCsn,
    ApexDepartmentId,
    LocationEpicId,
    EventType,
    EventDetail,
    EventSource,
    BillingCode,
    CodeType,
    ProcedureGroup,
    [Procedure],
    Amount,
    AmountType,
    BillingSystemType,
    ServiceDateKey,
    PatientDurableKey,
    ProviderEpicId,
    EncounterKey,
    Name,
    PatientFriendlyName,
    Category
)
SELECT *
FROM #temp t;



TRUNCATE TABLE MarketingWorkSpace.NoteExtraction.ProviderTreatmentCounts;
INSERT INTO MarketingWorkSpace.NoteExtraction.ProviderTreatmentCounts
SELECT t.ProviderEpicId,
       t.BillingCode,
       t.CodeType,
       t.ProcedureGroup,
       t.[Procedure],
       t.EventDetail AS ProcedureDetail,
       c.DescriptionsJson,
       COUNT(DISTINCT t.EncounterKey) ASMarketingWorkSpace.NoteExtraction.ProviderTreatmentCount EncounterCount,
       COUNT(DISTINCT t.PatientDurableKey) AS PatientCount,
       COUNT(DISTINCT t.ServiceDateKey) AS DistinctServiceDates,
       MAX(t.ServiceDateKey) AS MostRecentServiceDateKey

FROM MarketingWorkSpace.NoteExtraction.ProviderTreatmentDetails t
    LEFT OUTER JOIN #codes c
        ON t.BillingCode = c.BillingCodeMarketingWorkSpace.NoteExtraction.ProviderTreatmentCount


GROUP BY t.ProviderEpicId,
         t.BillingCode,
         t.CodeType,
         t.ProcedureGroup,
         t.[Procedure],
         c.DescriptionsJson,
         t.EventDetail;


