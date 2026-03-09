### EHR Preprocessing
Scripts for exporting and transforming the tabular EHR data for the MOTOR foundation model.

A Python CLI wrapper (`src/utils/bash_wrapper.py`) runs the bash scripts with the correct environment variables from `Config`.

```bash
# 1. Export cohort-only EHR data (subjects in mimiciv_ext.cohort)
python -m src.utils.bash_wrapper export-cohort

# 2. Export pretraining EHR data (all subjects EXCEPT test split)
python -m src.utils.bash_wrapper export-pretraining

# 3. Convert exported CSVs to MEDS format
python -m src.utils.bash_wrapper meds-pipeline cohort
python -m src.utils.bash_wrapper meds-pipeline pretraining

# 4. Generate prediction labels and anchor times for the cohort
python -m src.data.preprocess.ehr.ehr_labels
```