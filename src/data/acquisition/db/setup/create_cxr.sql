DROP SCHEMA IF EXISTS mimiciv_cxr CASCADE;
CREATE SCHEMA mimiciv_cxr;

DROP TABLE IF EXISTS mimiciv_cxr.record_list;
CREATE TABLE mimiciv_cxr.record_list
(
  subject_id INTEGER NOT NULL,
  study_id INTEGER NOT NULL,
  dicom_id VARCHAR(200),
  path VARCHAR(200)
);

DROP TABLE IF EXISTS mimiciv_cxr.study_list;
CREATE TABLE mimiciv_cxr.study_list (
    subject_id INTEGER NOT NULL,
    study_id INTEGER NOT NULL,
    path VARCHAR(200)
);

DROP TABLE IF EXISTS mimiciv_cxr.metadata;
CREATE TABLE mimiciv_cxr.metadata (
    dicom_id            TEXT,
    subject_id          BIGINT,
    study_id            BIGINT,
    PerformedProcedureStepDescription TEXT,
    ViewPosition        TEXT,
    Rows                INTEGER,
    Columns             INTEGER,
    StudyDate           INTEGER,
    StudyTime           DOUBLE PRECISION,
    ProcedureCodeSequence_CodeMeaning TEXT,
    ViewCodeSequence_CodeMeaning      TEXT,
    PatientOrientationCodeSequence_CodeMeaning TEXT
);