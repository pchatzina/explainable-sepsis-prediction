#!/bin/bash
# export_pretraining_data.sh
# Purpose: Exports MIMIC-IV data for pretraining.
# Logic: Exports ALL data (background subjects + train/val cohort), EXCLUDING ONLY the 'test' split subjects.
# Requirements: pigz installed, .pgpass configured

set -euo pipefail

if [ -z "${DB:-}" ] || [ -z "${BASE_OUTPUT_DIR:-}" ]; then
  echo "Error: Required environment variables DB or BASE_OUTPUT_DIR are missing."
  echo "This script must be run via the Python wrapper or with variables explicitly exported."
  exit 1
fi

# We use the table that contains the 'dataset_split' column
SPLIT_TABLE="mimiciv_ext.dataset_splits"

# Define Schemas to export (Output Folder Name -> Source DB Schema)
declare -A SCHEMAS
SCHEMAS["hosp"]="mimiciv_hosp"
SCHEMAS["icu"]="mimiciv_icu"

# --- Script Start ---

echo "Starting MIMIC-IV Pretraining Data Export..."
echo "Target DB: $DB"
echo "Output Dir: $BASE_OUTPUT_DIR"
echo "Excluding Subjects labeled 'test' in: $SPLIT_TABLE"

# Loop through the defined schemas
for key in "${!SCHEMAS[@]}"; do 
    output_folder_name=$key
    source_schema=${SCHEMAS[$key]}
    
    # Create output directory
    full_output_dir="${BASE_OUTPUT_DIR}/${output_folder_name}"
    mkdir -p "$full_output_dir"
    
    echo "--------------------------------------------------------"
    echo "Processing Schema: $source_schema (Output: $output_folder_name)"
    echo "--------------------------------------------------------"

    # Get list of tables directly from the SOURCE schema
    tables=$(psql -d "$DB" -t -A -c "SELECT tablename FROM pg_tables WHERE schemaname='$source_schema' ORDER BY tablename")

    for table in $tables; do
        echo -n "  → Exporting $table... "

        # 1. Check if the table has a 'subject_id' column
        has_subject_id=$(psql -d "$DB" -t -A -c "SELECT count(*) FROM information_schema.columns WHERE table_schema='$source_schema' AND table_name='$table' AND column_name='subject_id'")

        # 2. Construct the COPY query
        if [ "$has_subject_id" -eq "1" ]; then
            # Case A: Table has subject_id
            # EXPORT LOGIC: Include everyone EXCEPT those marked as 'test' in the split table
            # - Non-cohort subjects (not in table) -> Included (NOT EXISTS is true)
            # - Train/Val subjects (in table, split!='test') -> Included (NOT EXISTS is true)
            # - Test subjects (in table, split='test') -> Excluded (NOT EXISTS is false)
            
            SQL_QUERY="COPY (SELECT t.* FROM ${source_schema}.${table} t WHERE NOT EXISTS (SELECT 1 FROM ${SPLIT_TABLE} s WHERE s.subject_id = t.subject_id AND s.dataset_split = 'test')) TO STDOUT CSV HEADER"
            echo -n "[Filtered: Excluding 'Test' Split Only] "
        else
            # Case B: Dictionary / No subject_id
            # EXPORT LOGIC: Full Table Export (no filtering)
            SQL_QUERY="COPY ${source_schema}.${table} TO STDOUT CSV HEADER"
            echo -n "[Full Export: Dictionary] "
        fi

        # 3. Execute Copy -> Pipe to Pigz -> Write to Disk
        set +o pipefail
        psql -d "$DB" -c "$SQL_QUERY" | pigz > "${full_output_dir}/${table}.csv.gz"
        pipe_status=("${PIPESTATUS[@]}")
        set -o pipefail

        if [ "${pipe_status[0]}" -eq 0 ] && [ "${pipe_status[1]}" -eq 0 ]; then
             echo "✔ Done."
        else
             echo "✘ ERROR (psql=${pipe_status[0]}, pigz=${pipe_status[1]})."
             exit 1
        fi
    done
done

echo "--------------------------------------------------------"
echo "All exports completed successfully."
