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
    bandwidth TEXT,
    filtering TEXT,
    rr_interval NUMERIC,
    p_onset NUMERIC,
    p_end NUMERIC,
    qrs_onset NUMERIC,
    qrs_end NUMERIC,
    t_end NUMERIC,
    p_axis NUMERIC,
    qrs_axis NUMERIC,
    t_axis NUMERIC
);