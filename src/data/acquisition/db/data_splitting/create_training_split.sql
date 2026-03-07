-- ------------------------------------------------------------------
-- Step 1: Define the "Modality Signature" and "Freshness" for every patient
-- ------------------------------------------------------------------
DROP TABLE IF EXISTS mimiciv_ext.patient_strata;
CREATE TABLE mimiciv_ext.patient_strata AS
SELECT 
    c.subject_id,
    c.hadm_id,
    c.sepsis_label,
    -- 1. Modality Signature
    CONCAT(
        'EHR',
        CASE WHEN cxr.hadm_id IS NOT NULL THEN '_CXR' ELSE '' END,
        CASE WHEN ecg.hadm_id IS NOT NULL THEN '_ECG' ELSE '' END
    ) AS modality_signature,
    
    -- 2. CXR Freshness Tier (Time gap between prediction point and image)
    CASE 
        WHEN cxr.hadm_id IS NULL THEN 'N/A'
        WHEN EXTRACT(EPOCH FROM (c.anchor_time - cxr.study_timestamp)) / 3600.0 <= 12 THEN 'Fresh (0-12h)'
        WHEN EXTRACT(EPOCH FROM (c.anchor_time - cxr.study_timestamp)) / 3600.0 <= 24 THEN 'Recent (12-24h)'
        ELSE 'Older (>24h)'
    END AS cxr_freshness_tier,

    -- 3. ECG Freshness Tier
    CASE 
        WHEN ecg.hadm_id IS NULL THEN 'N/A'
        WHEN EXTRACT(EPOCH FROM (c.anchor_time - ecg.study_timestamp)) / 3600.0 <= 12 THEN 'Fresh (0-12h)'
        WHEN EXTRACT(EPOCH FROM (c.anchor_time - ecg.study_timestamp)) / 3600.0 <= 24 THEN 'Recent (12-24h)'
        ELSE 'Older (>24h)'
    END AS ecg_freshness_tier

FROM mimiciv_ext.cohort c
LEFT JOIN mimiciv_ext.cohort_cxr cxr 
    ON c.subject_id = cxr.subject_id AND c.hadm_id = cxr.hadm_id
LEFT JOIN mimiciv_ext.cohort_ecg ecg 
    ON c.subject_id = ecg.subject_id AND c.hadm_id = ecg.hadm_id;

-- ------------------------------------------------------------------
-- Step 2: Assign Split Labels (Train/Val/Test)
-- Stratified by Sepsis Label, Modality Signature, CXR Freshness, AND ECG Freshness
-- ------------------------------------------------------------------
DROP TABLE IF EXISTS mimiciv_ext.dataset_splits;
CREATE TABLE mimiciv_ext.dataset_splits AS
WITH randomized AS (
    SELECT 
        subject_id,
        hadm_id,
        modality_signature,
        sepsis_label,
        cxr_freshness_tier,
        ecg_freshness_tier,
        -- Deterministic sorting partitioned by ALL strata
        ROW_NUMBER() OVER (
            PARTITION BY modality_signature, sepsis_label, cxr_freshness_tier, ecg_freshness_tier 
            ORDER BY md5(subject_id::text)
        ) as rn,
        COUNT(*) OVER (
            PARTITION BY modality_signature, sepsis_label, cxr_freshness_tier, ecg_freshness_tier
        ) as total_in_group
    FROM mimiciv_ext.patient_strata
)
SELECT 
    subject_id,
    hadm_id,
    modality_signature,
    sepsis_label,
    cxr_freshness_tier,
    ecg_freshness_tier,
    CASE 
        WHEN rn <= (total_in_group * 0.70) THEN 'train'
        WHEN rn <= (total_in_group * 0.85) THEN 'validate'
        ELSE 'test'
    END AS dataset_split
FROM randomized;
