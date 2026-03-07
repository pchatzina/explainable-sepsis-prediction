-- ------------------------------------------------------------------
-- Script: create_final_cohort.sql
-- Purpose: Finalizes the cohort by handling duplicates and selecting single modalities.
--          - Rule 1: One admission per patient (Priority: Has Modality > Earliest Time).
--          - Rule 2: One CXR/ECG per admission (Priority: Latest timestamp).
-- ------------------------------------------------------------------

-- ==================================================================
-- 1. IDENTIFY THE SINGLE "BEST" ADMISSION PER PATIENT
-- ==================================================================
DROP TABLE IF EXISTS mimiciv_ext.kept_admissions_temp;

CREATE TEMP TABLE kept_admissions_temp AS
WITH has_modality AS (
    SELECT 
        s.subject_id, 
        s.hadm_id, 
        s.admittime,
        -- Priority Flag: 1 if the admission has EITHER CXR OR ECG, else 0
        CASE 
            WHEN c.hadm_id IS NOT NULL OR e.hadm_id IS NOT NULL THEN 1 
            ELSE 0 
        END AS has_modality
    FROM mimiciv_ext.generic_ehr_cohort s
    LEFT JOIN (SELECT DISTINCT hadm_id FROM mimiciv_ext.generic_cxr_cohort) c 
        ON s.hadm_id = c.hadm_id
    LEFT JOIN (SELECT DISTINCT hadm_id FROM mimiciv_ext.generic_ecg_cohort) e 
        ON s.hadm_id = e.hadm_id
),
ranked AS (
    SELECT 
        subject_id, 
        hadm_id, 
        -- Rank 1: Has Modality (DESC), Rank 2: Oldest Admission (ASC)
        ROW_NUMBER() OVER (
            PARTITION BY subject_id 
            ORDER BY has_modality DESC, admittime ASC, hadm_id ASC
        ) AS rn
    FROM has_modality
)
SELECT subject_id, hadm_id 
FROM ranked 
WHERE rn = 1;

-- ==================================================================
-- 2. CREATE FINAL EHR COHORT (Unique Subjects)
-- ==================================================================
DROP TABLE IF EXISTS mimiciv_ext.cohort;

CREATE TABLE mimiciv_ext.cohort AS
SELECT s.*
FROM mimiciv_ext.generic_ehr_cohort s
JOIN kept_admissions_temp k 
    ON s.subject_id = k.subject_id AND s.hadm_id = k.hadm_id;

-- PK & Index
ALTER TABLE mimiciv_ext.cohort ADD PRIMARY KEY (subject_id);
CREATE INDEX idx_cohort_hadm ON mimiciv_ext.cohort(hadm_id);

-- ==================================================================
-- 3. CREATE FINAL CXR COHORT (Single Latest Study)
-- ==================================================================
DROP TABLE IF EXISTS mimiciv_ext.cohort_cxr;

CREATE TABLE mimiciv_ext.cohort_cxr AS
SELECT DISTINCT ON (c.subject_id)
    c.id,
    c.subject_id,
    c.hadm_id,
    c.study_id,
    c.study_timestamp,
    c.study_path
FROM mimiciv_ext.generic_cxr_cohort c
JOIN kept_admissions_temp k 
    ON c.subject_id = k.subject_id AND c.hadm_id = k.hadm_id
ORDER BY 
    c.subject_id, 
    c.study_timestamp DESC,
    c.study_id DESC;

-- PK, FK, & Index
ALTER TABLE mimiciv_ext.cohort_cxr ADD PRIMARY KEY (id);
CREATE INDEX idx_cxr_subject ON mimiciv_ext.cohort_cxr(subject_id);
ALTER TABLE mimiciv_ext.cohort_cxr 
    ADD CONSTRAINT fk_cxr_cohort FOREIGN KEY (subject_id) 
    REFERENCES mimiciv_ext.cohort (subject_id) ON DELETE CASCADE;

-- ==================================================================
-- 4. CREATE FINAL ECG COHORT (Single Latest Study)
-- ==================================================================
DROP TABLE IF EXISTS mimiciv_ext.cohort_ecg;

CREATE TABLE mimiciv_ext.cohort_ecg AS
SELECT DISTINCT ON (e.subject_id)
    e.*
FROM mimiciv_ext.generic_ecg_cohort e
JOIN kept_admissions_temp k 
    ON e.subject_id = k.subject_id AND e.hadm_id = k.hadm_id
ORDER BY 
    e.subject_id, 
    e.study_timestamp DESC,
    e.study_id DESC;

-- PK, FK, & Index
ALTER TABLE mimiciv_ext.cohort_ecg ADD PRIMARY KEY (id);
CREATE INDEX idx_ecg_subject ON mimiciv_ext.cohort_ecg(subject_id);
ALTER TABLE mimiciv_ext.cohort_ecg 
    ADD CONSTRAINT fk_ecg_cohort FOREIGN KEY (subject_id) 
    REFERENCES mimiciv_ext.cohort (subject_id) ON DELETE CASCADE;

-- Cleanup
DROP TABLE IF EXISTS kept_admissions_temp;