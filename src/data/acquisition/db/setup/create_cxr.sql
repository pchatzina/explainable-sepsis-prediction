-- ===================================================================
-- Script: create_cxr.sql
-- ===================================================================
-- Purpose: Create mimiciv_cxr schema and tables for DICOM metadata.
--
-- Schema: mimiciv_cxr
--   Scope: Metadata for ALL MIMIC-IV CXR studies (not just cohort)
--   Usage: Queries in Phase 3 (create_generic_modalities_cohort.sql)
--
-- Tables:
--   1. record_list: One row per DICOM file
--   2. study_list: One row per CXR study (may contain multiple DICOM records per study)
--   3. metadata: One row per DICOM file
-- ===================================================================

DROP SCHEMA IF EXISTS mimiciv_cxr CASCADE;
CREATE SCHEMA mimiciv_cxr;

-- Table 1: CXR record list (DICOM file mapping)
-- Populated from cxr-record-list.csv.gz (PhysioNet download)
DROP TABLE IF EXISTS mimiciv_cxr.record_list;
CREATE TABLE mimiciv_cxr.record_list
(
  subject_id INTEGER NOT NULL,
  study_id INTEGER NOT NULL,
  dicom_id VARCHAR(200),
  path VARCHAR(200)
);

-- Table 2: CXR study list (CXR study mapping)
-- Populated from cxr-study-list.csv.gz (PhysioNet download)
DROP TABLE IF EXISTS mimiciv_cxr.study_list;
CREATE TABLE mimiciv_cxr.study_list (
    subject_id INTEGER NOT NULL,
    study_id INTEGER NOT NULL,
    path VARCHAR(200)
);

-- Table 3: CXR DICOM metadata (detailed imaging metadata)
-- Populated from mimic-cxr-2.0.0-metadata.csv.gz (PhysioNet download)
-- Used in Phase 3 for image selection: ViewPosition > Resolution priority
DROP TABLE IF EXISTS mimiciv_cxr.metadata;
CREATE TABLE mimiciv_cxr.metadata (
    dicom_id            TEXT,
    subject_id          BIGINT,
    study_id            BIGINT,
    PerformedProcedureStepDescription TEXT,
    ViewPosition        TEXT,                    -- Priority: PA > AP > LATERAL
    Rows                INTEGER,                -- Image height (pixels)
    Columns             INTEGER,                -- Image width (pixels)
    StudyDate           INTEGER,                -- YYYYMMDD format (for timestamp reconstruction)
    StudyTime           DOUBLE PRECISION,       -- HHMMSS format (fractional seconds)
    ProcedureCodeSequence_CodeMeaning TEXT,    -- Procedure type (e.g., "CHEST PA/LATERAL")
    ViewCodeSequence_CodeMeaning      TEXT,    -- View name (redundant with ViewPosition)
    PatientOrientationCodeSequence_CodeMeaning TEXT -- Patient position
);