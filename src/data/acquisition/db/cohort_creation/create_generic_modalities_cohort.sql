-- ===================================================================
-- Script: create_generic_modalities_cohort.sql
-- ===================================================================
-- Purpose: Identifies available modalities (CXR, ECG) for the generic cohort.
--          Filters to images/signals within clinical causality window and applies
--          modality-specific selection rules (best quality per study).
--
-- Modality Availability: 66-hour pre-anchor window
--   Rationale:
--     - Ensures all modality data is acquired at most 72 hours before sepsis onset 
--       (anchor_time = sepsis_onset - 6h).
--
-- CXR Selection Rules (Priority, in order):
--   1. ViewPosition: PA > AP > LATERAL/LL > OTHER
--   2. Resolution: Highest pixel count (rows * columns) preferred
--   3. Ordering: Earliest dicom_id
--
-- ECG Selection: All valid records in window kept (no deduplication; Phase 3 selects latest)
--
-- Output:
--   - mimiciv_ext.generic_cxr_cohort: CXR metadata
--   - mimiciv_ext.generic_ecg_cohort: ECG metadata
-- ===================================================================

-- ==================================================================
-- 1. CHEST X-RAY (CXR) COHORT
-- ==================================================================
DROP TABLE IF EXISTS mimiciv_ext.generic_cxr_cohort;

CREATE TABLE mimiciv_ext.generic_cxr_cohort (
    id SERIAL PRIMARY KEY,
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    study_id INTEGER NOT NULL,
    study_timestamp TIMESTAMP WITHOUT TIME ZONE,
    study_path VARCHAR(255),
    dicom_id VARCHAR(255),
    rows INTEGER,
    columns INTEGER,
    view_position TEXT
);

-- Ensure unique study per admission (prevents duplicate rows in downstream joins)
ALTER TABLE mimiciv_ext.generic_cxr_cohort
ADD CONSTRAINT unique_cxr_study_per_admission UNIQUE (subject_id, hadm_id, study_id);

-- Query and insert CXR records: join cohort + DICOM metadata + filter by window
-- DISTINCT ON (subject_id, study_id) picks best image per study via ORDER BY rules
INSERT INTO mimiciv_ext.generic_cxr_cohort
    (subject_id, hadm_id, study_id, study_timestamp, study_path, dicom_id, rows, columns, view_position)
SELECT DISTINCT ON (s.subject_id, s.study_id)
    s.subject_id,
    c.hadm_id,
    s.study_id,
    -- Reconstruct timestamp from DICOM integer fields (YYYYMMDD, HHMMSS format)
    make_timestamp(
        m.studydate / 10000,
        (m.studydate / 100) % 100,
        m.studydate % 100,
        floor(m.studytime / 10000)::integer,
        floor(m.studytime / 100)::integer % 100,
        floor(m.studytime)::integer % 100
    ) AS study_timestamp,
    -- Convert DICOM path from .dcm to .jpg (image storage format)
    regexp_replace(r.path, '\.dcm$', '.jpg') AS study_path,
    r.dicom_id,
    m.rows,
    m.columns,
    m.viewposition
FROM mimiciv_cxr.study_list s
JOIN mimiciv_ext.generic_ehr_cohort c ON s.subject_id = c.subject_id
JOIN mimiciv_cxr.record_list r ON s.study_id = r.study_id AND s.subject_id = r.subject_id
JOIN mimiciv_cxr.metadata m ON r.dicom_id = m.dicom_id
WHERE 
    -- WINDOW: anchor_time - 66h
    -- Lower bound: image must be >= admission time
    make_timestamp(
        m.studydate / 10000,
        (m.studydate / 100) % 100,
        m.studydate % 100,
        floor(m.studytime / 10000)::integer,
        floor(m.studytime / 100)::integer % 100,
        floor(m.studytime)::integer % 100
    ) >= c.admittime
    AND make_timestamp(
        m.studydate / 10000,
        (m.studydate / 100) % 100,
        m.studydate % 100,
        floor(m.studytime / 10000)::integer,
        floor(m.studytime / 100)::integer % 100,
        floor(m.studytime)::integer % 100
    ) >= c.anchor_time - INTERVAL '66 hours'
    AND make_timestamp(
        m.studydate / 10000,
        (m.studydate / 100) % 100,
        m.studydate % 100,
        floor(m.studytime / 10000)::integer,
        floor(m.studytime / 100)::integer % 100,
        floor(m.studytime)::integer % 100
    ) <= c.anchor_time
ORDER BY 
    s.subject_id, 
    s.study_id,
    -- SELECTION RULE 1: ViewPosition Priority (PA > AP > LATERAL > OTHER)
    CASE 
        WHEN m.ViewPosition = 'PA' THEN 1 
        WHEN m.ViewPosition = 'AP' THEN 2 
        WHEN m.ViewPosition IN ('LATERAL', 'LL') THEN 3 
        ELSE 4
    END ASC,
    -- SELECTION RULE 2: Resolution Priority (Highest pixels first)
    (m.Rows * m.Columns) DESC,
    r.dicom_id ASC
ON CONFLICT ON CONSTRAINT unique_cxr_study_per_admission DO NOTHING;


-- ==================================================================
-- 2. ECG COHORT
-- ==================================================================
DROP TABLE IF EXISTS mimiciv_ext.generic_ecg_cohort;

CREATE TABLE mimiciv_ext.generic_ecg_cohort (
    id SERIAL PRIMARY KEY,
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    study_id INTEGER NOT NULL, 
    study_timestamp TIMESTAMP WITHOUT TIME ZONE,
    study_path VARCHAR(255)
);

-- Query and insert ECG records: join cohort + ECG metadata + filter by causality window
-- Note: Multiple ECGs per patient may be present; Phase 3 (create_final_cohort.sql)
-- will select the latest one per subject for the final cohort.
INSERT INTO mimiciv_ext.generic_ecg_cohort (
    subject_id,
    hadm_id,
    study_id,
    study_timestamp,
    study_path
)
SELECT
    t1.subject_id,
    t1.hadm_id,
    t2.study_id,
    t2.ecg_time AS study_timestamp,
    t2.path AS study_path
FROM
    mimiciv_ext.generic_ehr_cohort t1
JOIN
    mimiciv_ecg.record_list t2
    ON t1.subject_id = t2.subject_id
WHERE
    -- WINDOW: anchor_time - 66h
    -- Lower bound: ECG >= admission
    t2.ecg_time >= t1.admittime
    -- Upper bound: ECG within 66h pre-anchor
    AND t2.ecg_time >= t1.anchor_time - INTERVAL '66 hours'
    -- Upper bound: ECG <= anchor_time (no future information)
    AND t2.ecg_time <= t1.anchor_time;