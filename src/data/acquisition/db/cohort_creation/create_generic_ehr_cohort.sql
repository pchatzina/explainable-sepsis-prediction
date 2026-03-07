-- ------------------------------------------------------------------
-- Script: create_generic_ehr_cohort.sql
-- Purpose: Creates the initial cohort of Sepsis Positive and Negative patients.
--          - Positives: Sepsis onset >= 24h after admission. Anchor = Onset - 6h.
--          - Negatives: No sepsis history. Anchor = ICU Intime + Median_Time_To_Sepsis - 6h.
--          - Note: This table may contain multiple admissions per subject.
--            Subject-level deduplication happens in a later step based on modality availability.
-- ------------------------------------------------------------------

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
    anchor_time TIMESTAMP(0) WITHOUT TIME ZONE NOT NULL, -- Prediction Time (Onset/Pseudo-Onset - 6h)
    sepsis_onset TIMESTAMP(0) WITHOUT TIME ZONE,         -- Sepsis onset time (NULL for negatives)

    -- Outcome
    sepsis_label SMALLINT NOT NULL                       -- 1 for Positive, 0 for Negative
);

WITH
-- 1. IDENTIFY ALL SEPSIS SUBJECTS FOR EXCLUSION
--    We filter out subjects who have EVER had sepsis (strict control group).
all_sepsis_subjects AS (
    SELECT DISTINCT subject_id
    FROM mimiciv_derived.sepsis3
),

-- 2. POSITIVE COHORT DEFINITION
--    Patients who developed sepsis at least 24 hours after hospital admission.
sepsis_positive AS (
    SELECT
        a.subject_id,
        s.stay_id,
        i.hadm_id,
        i.intime AS icu_intime,
        a.admittime,
        -- Sepsis onset is the latest of suspected infection and SOFA score >= 2
        GREATEST(s.suspected_infection_time, s.sofa_time) AS onset_time
    FROM
        mimiciv_derived.sepsis3 s
    JOIN
        mimiciv_icu.icustays i ON s.stay_id = i.stay_id
    JOIN
        mimiciv_hosp.admissions a ON a.hadm_id = i.hadm_id
    WHERE
        GREATEST(s.suspected_infection_time, s.sofa_time) >= a.admittime + INTERVAL '24 hours'
),

-- 3. CALCULATE DYNAMIC MEDIAN TIME (in hours)
--    Calculates the median time from ICU admission to Sepsis Onset for positive cases.
median_hours AS (
    SELECT
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (sp.onset_time - sp.icu_intime)) / 3600.0) AS median_hours_to_sepsis
    FROM
        sepsis_positive sp
),

-- 4. PROCESS POSITIVE PATIENTS (Select First ICU Stay per Admission)
first_icu_sepsis AS (
    SELECT
        sp.subject_id,
        sp.hadm_id,
        sp.stay_id,
        sp.icu_intime,
        sp.onset_time AS sepsis_onset,
        -- Anchor time: 6 hours before official onset
        sp.onset_time - INTERVAL '6 hours' AS anchor_time,
        -- Rank ICU stays to pick the first one per admission
        ROW_NUMBER() OVER (PARTITION BY sp.hadm_id ORDER BY sp.icu_intime) AS rn
    FROM
        sepsis_positive sp
),

-- 5. NEGATIVE COHORT CANDIDATES
--    Patients without sepsis history, with sufficient length of stay.
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
        -- Anchor time: (ICU Intime + Median Time) - 6 hours
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
        -- Exclude any subject who appears in the sepsis3 table
        s2.subject_id IS NULL
        -- Ensure patient is still in ICU/Hospital at the calculated pseudo-onset time
        AND (i.intime + (m.median_hours_to_sepsis * INTERVAL '1 hour')) <= LEAST(i.outtime, a.dischtime)
        -- Ensure Anchor Time is at least 18h after admission (Consistency with Positive >24h onset - 6h prediction window)
        AND ((i.intime + (m.median_hours_to_sepsis * INTERVAL '1 hour')) - INTERVAL '6 hours') >= a.admittime + INTERVAL '18 hours'
),

-- 6. PROCESS NEGATIVE PATIENTS (Select First ICU Stay per Admission)
first_icu_negative AS (
    SELECT
        nc.*,
        ROW_NUMBER() OVER (PARTITION BY nc.hadm_id ORDER BY nc.icu_intime) AS rn
    FROM
        negative_candidates nc
)

-- 7. FINAL INSERTION
INSERT INTO mimiciv_ext.generic_ehr_cohort (
    subject_id, hadm_id, stay_id,
    admittime, icu_intime,
    anchor_time, sepsis_onset, sepsis_label
)
-- Insert Positives (Only the first ICU stay per admission)
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

-- Insert Negatives (Only the first ICU stay per admission)
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