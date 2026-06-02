-- ===================================================================
-- Script: create_ecg.sql
-- ===================================================================
-- Purpose: Create mimiciv_ecg schema and tables for ECG metadata.
--
-- Schema: mimiciv_ecg
--   Scope: Metadata for ALL MIMIC-IV ECG studies (not just cohort)
--   Usage: Queries in Phase 3 (create_generic_modalities_cohort.sql)
--
-- Tables:
--   1. record_list: One row per ECG recording
--   2. machine_measurements: One row per ECG recording (computed ECG features)
-- ===================================================================

DROP SCHEMA IF EXISTS mimiciv_ecg CASCADE;
CREATE SCHEMA mimiciv_ecg;

DROP TABLE IF EXISTS mimiciv_ecg.record_list;
CREATE TABLE mimiciv_ecg.record_list
(
  subject_id INTEGER NOT NULL,
  study_id INTEGER NOT NULL,
  file_name INTEGER NOT NULL,
  ecg_time TIMESTAMP NOT NULL,
  path VARCHAR(200)
);

DROP TABLE IF EXISTS mimiciv_ecg.machine_measurements;
CREATE TABLE mimiciv_ecg.machine_measurements (
    subject_id INTEGER NOT NULL,
    study_id INTEGER NOT NULL,
    cart_id INTEGER NOT NULL,
    ecg_time TIMESTAMP WITHOUT TIME ZONE,
    
    -- Machine interpretation reports (18 categories of ECG analysis)
    -- report_0 through report_17
    -- Used for optional automated ECG screening in preprocessing (Phase 2)
    report_0 TEXT,
    report_1 TEXT,
    report_2 TEXT,
    report_3 TEXT,
    report_4 TEXT,
    report_5 TEXT,
    report_6 TEXT,
    report_7 TEXT,
    report_8 TEXT,
    report_9 TEXT,
    report_10 TEXT,
    report_11 TEXT,
    report_12 TEXT,
    report_13 TEXT,
    report_14 TEXT,
    report_15 TEXT,
    report_16 TEXT,
    report_17 TEXT,
    
    -- Signal quality indicators
    bandwidth TEXT,                   -- Frequency bandwidth
    filtering TEXT,                   -- Filtering applied
    
    -- Cardiac interval measurements
    rr_interval NUMERIC,              -- RR interval
    p_onset NUMERIC,                  -- P wave onset
    p_end NUMERIC,                    -- P wave end
    qrs_onset NUMERIC,                -- QRS complex onset
    qrs_end NUMERIC,                  -- QRS complex end
    t_end NUMERIC,                    -- T wave end
    
    -- ECG axis measurements
    p_axis NUMERIC,                   -- P wave axis
    qrs_axis NUMERIC,                 -- QRS axis
    t_axis NUMERIC                    -- T wave axis
);