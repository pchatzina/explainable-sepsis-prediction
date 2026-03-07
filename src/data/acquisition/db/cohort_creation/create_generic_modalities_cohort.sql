-- ------------------------------------------------------------------
-- Script: create_generic_modalities_cohort.sql
-- Purpose: Identifies available modalites (CXR, ECG) for the generic cohort.
--          - Filters modalities that occurred BEFORE the Anchor Time 
--            AND within 66 hours prior to the Anchor Time 
--            AND after Admission.
--          - CXR: Selects the "best" image per study based on ViewPosition and Resolution.
--          - ECG: Selects all valid ECGs within the window.
-- ------------------------------------------------------------------

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

-- Ensure unique study per admission mapping
ALTER TABLE mimiciv_ext.generic_cxr_cohort
ADD CONSTRAINT unique_cxr_study_per_admission UNIQUE (subject_id, hadm_id, study_id);

INSERT INTO mimiciv_ext.generic_cxr_cohort
    (subject_id, hadm_id, study_id, study_timestamp, study_path, dicom_id, rows, columns, view_position)
SELECT DISTINCT ON (s.subject_id, s.study_id)
    s.subject_id,
    c.hadm_id,
    s.study_id,
    -- Construct Timestamp from integer dates/times
    make_timestamp(
        m.studydate / 10000,
        (m.studydate / 100) % 100,
        m.studydate % 100,
        floor(m.studytime / 10000)::integer,
        floor(m.studytime / 100)::integer % 100,
        floor(m.studytime)::integer % 100
    ) AS study_timestamp,
    -- Replace .dcm extension with .jpg
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
    -- Filter 1: Time Window (Must be between Admission and Anchor Time, and max 66h pre-anchor) 
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
    -- Filter 2: Priority Rule for Multiple Images per Study
    -- 1. View Position Priority (PA > AP > Lateral)
    CASE 
        WHEN m.ViewPosition = 'PA' THEN 1 
        WHEN m.ViewPosition = 'AP' THEN 2 
        WHEN m.ViewPosition IN ('LATERAL', 'LL') THEN 3 
        ELSE 4
    END ASC,
    -- 2. Resolution Priority (Highest pixels first)
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
    -- Time Window Filter (After admit, before anchor, max 66h pre-anchor)
    t2.ecg_time >= t1.admittime
    AND t2.ecg_time >= t1.anchor_time - INTERVAL '66 hours'
    AND t2.ecg_time <= t1.anchor_time;