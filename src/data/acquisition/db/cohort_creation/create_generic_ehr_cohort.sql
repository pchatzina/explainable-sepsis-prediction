-- ===================================================================
-- Script: create_generic_ehr_cohort.sql
-- ===================================================================
-- Purpose: Defines the initial cohort of Sepsis Positive and Negative cases.
--
-- Clinical Definitions:
--   POSITIVES: Patients diagnosed with sepsis >= 24 hours post-hospital admission.
--     Anchor Time = Sepsis Onset - 6 hours (prediction window boundary).
--     Rationale: 24h delay prevents early admissions of already-septic patients;
--               6h pre-onset window ensures no future data leakage into prediction.
--
--   NEGATIVES: Patients WITHOUT ANY sepsis history (strict control group).
--     Anchor Time = ICU_intime + Median_Time_To_Sepsis - 6 hours.
--     Rationale: Uses median(sepsis onset time) from positives as a pseudo-onset.
--               This ensures negatives have the same observational window length
--               as positives (temporal distribution matching).
--               6h pre-pseudo-onset ensures consistency with positive anchoring.
--
-- Critical Implementation Notes (Data Leakage Prevention):
--   1. All clinical features (vitals, labs, treatments) used for prediction MUST be
--      acquired from [admission, anchor_time] interval ONLY.
--   2. Modality data (CXR, ECG) will be filtered to a 66-hour pre-anchor window
--      in Phase 3 (create_generic_modalities_cohort.sql).
--   3. First ICU stay per admission is selected (deduplicated in Phase 3 by modality).
--
-- Output:
--   Table: mimiciv_ext.generic_ehr_cohort
--   Note: May contain multiple admissions per subject (deduplicated later by modality).
-- ===================================================================

DROP SCHEMA IF EXISTS mimiciv_ext CASCADE;
CREATE SCHEMA mimiciv_ext;

DROP TABLE IF EXISTS mimiciv_ext.generic_ehr_cohort;
CREATE TABLE mimiciv_ext.generic_ehr_cohort (
    -- Patient Identifiers
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    stay_id INT NOT NULL,

    -- Time Points
    admittime TIMESTAMP(0) WITHOUT TIME ZONE NOT NULL,   -- Hospital Admission Time
    icu_intime TIMESTAMP(0) WITHOUT TIME ZONE NOT NULL,  -- ICU Admission Time
    anchor_time TIMESTAMP(0) WITHOUT TIME ZONE NOT NULL, -- Prediction Boundary: Sepsis Onset - 6h (or pseudo-onset - 6h)
    sepsis_onset TIMESTAMP(0) WITHOUT TIME ZONE,         -- Sepsis onset time (NULL for negatives)

    -- Outcome
    sepsis_label SMALLINT NOT NULL                       -- 1 for Positive, 0 for Negative
);

WITH
-- 1. IDENTIFY ALL SEPSIS SUBJECTS (FOR EXCLUSION FROM NEGATIVES)
-- Strict control group: must have NO sepsis history ever recorded
all_sepsis_subjects AS (
    SELECT DISTINCT subject_id
    FROM mimiciv_derived.sepsis3
),

-- 2. POSITIVE COHORT: Sepsis Onset >= 24h Post-Admission
-- Inclusion: GREATEST(suspected_infection_time, sofa_time) >= admittime + 24 hours
-- This ensures we have sufficient pre-sepsis observation window; early admissions
-- already in septic shock are excluded.
sepsis_positive AS (
    SELECT
        a.subject_id,
        s.stay_id,
        i.hadm_id,
        i.intime AS icu_intime,
        a.admittime,
        -- Sepsis3 protocol: onset is latest of infection detection + SOFA threshold
        GREATEST(s.suspected_infection_time, s.sofa_time) AS onset_time
    FROM
        mimiciv_derived.sepsis3 s
    JOIN
        mimiciv_icu.icustays i ON s.stay_id = i.stay_id
    JOIN
        mimiciv_hosp.admissions a ON a.hadm_id = i.hadm_id
    WHERE
        -- Enforce 24h minimum time-to-onset (inclusion criterion)
        GREATEST(s.suspected_infection_time, s.sofa_time) >= a.admittime + INTERVAL '24 hours'
),

-- 3. COMPUTE MEDIAN TIME-TO-SEPSIS (Distribution Matching for Negatives)
-- Negatives will use this as pseudo-onset to match positive temporal distribution
median_hours AS (
    SELECT
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (sp.onset_time - sp.icu_intime)) / 3600.0) AS median_hours_to_sepsis
    FROM
        sepsis_positive sp
),

-- 4. POSITIVE COHORT PROCESSING: Select First ICU Stay per Admission
-- Multiple ICU stays within same admission can occur. Take first (lowest icu_intime).
first_icu_sepsis AS (
    SELECT
        sp.subject_id,
        sp.hadm_id,
        sp.stay_id,
        sp.icu_intime,
        sp.onset_time AS sepsis_onset,
        -- ANCHOR: 6 hours before sepsis onset (data leakage prevention)
        sp.onset_time - INTERVAL '6 hours' AS anchor_time,
        -- Rank to select first ICU stay per admission
        ROW_NUMBER() OVER (PARTITION BY sp.hadm_id ORDER BY sp.icu_intime) AS rn
    FROM
        sepsis_positive sp
),

-- 5. NEGATIVE COHORT CANDIDATES: Strict Non-Sepsis Control Group
-- Inclusion criteria:
--   a) Never appears in sepsis3 table (s2.subject_id IS NULL)
--   b) Pseudo-onset time is within patient's ICU/Hospital stay
--   c) Anchor time (pseudo-onset - 6h) is >= 18h post-admission
--      (ensures minimum window length similar to positives: 24h - 6h = 18h)
negative_candidates AS (
    SELECT
        a.subject_id,
        i.hadm_id,
        i.stay_id,
        i.intime AS icu_intime,
        i.outtime AS icu_outtime,
        a.dischtime,
        a.admittime,
        m.median_hours_to_sepsis,
        -- ANCHOR: (ICU intime + Median Time) - 6h (pseudo-onset anchoring, matching positive strategy)
        (i.intime + (m.median_hours_to_sepsis * INTERVAL '1 hour')) - INTERVAL '6 hours' AS anchor_time
    FROM
        mimiciv_icu.icustays i
    JOIN
        mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
    CROSS JOIN
        median_hours m
    LEFT JOIN
        all_sepsis_subjects s2 ON a.subject_id = s2.subject_id
    WHERE
        -- Strict control: No sepsis history
        s2.subject_id IS NULL
        -- Ensure pseudo-onset is within patient's stay (no censoring/future dates)
        AND (i.intime + (m.median_hours_to_sepsis * INTERVAL '1 hour')) <= LEAST(i.outtime, a.dischtime)
        -- Ensure observation window is sufficient (minimum 18h, matching positive >= 24h - 6h)
        AND ((i.intime + (m.median_hours_to_sepsis * INTERVAL '1 hour')) - INTERVAL '6 hours') >= a.admittime + INTERVAL '18 hours'
),

-- 6. NEGATIVE COHORT PROCESSING: Select First ICU Stay per Admission
-- Same deduplication as positives: take first (lowest icu_intime) per admission
first_icu_negative AS (
    SELECT
        nc.*,
        ROW_NUMBER() OVER (PARTITION BY nc.hadm_id ORDER BY nc.icu_intime) AS rn
    FROM
        negative_candidates nc
)

-- 7. FINAL INSERTION: Union Positives + Negatives
INSERT INTO mimiciv_ext.generic_ehr_cohort (
    subject_id, hadm_id, stay_id,
    admittime, icu_intime,
    anchor_time, sepsis_onset, sepsis_label
)
-- Insert Positives (First ICU stay per admission only; rn = 1)
SELECT
    fis.subject_id,
    fis.hadm_id,
    fis.stay_id,
    a.admittime,
    fis.icu_intime,
    fis.anchor_time,
    fis.sepsis_onset,
    1 AS sepsis_label
FROM first_icu_sepsis fis
JOIN mimiciv_hosp.admissions a ON fis.hadm_id = a.hadm_id
WHERE fis.rn = 1

UNION ALL

-- Insert Negatives (First ICU stay per admission only; rn = 1)
SELECT
    fin.subject_id,
    fin.hadm_id,
    fin.stay_id,
    a.admittime,
    fin.icu_intime,
    fin.anchor_time,
    NULL AS sepsis_onset,
    0 AS sepsis_label
FROM first_icu_negative fin
JOIN mimiciv_hosp.admissions a ON fin.hadm_id = a.hadm_id
WHERE fin.rn = 1;