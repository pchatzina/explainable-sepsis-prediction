#!/bin/bash
# export_cohort_data.sh
# Purpose: Exports MIMIC-IV data directly from source, filtering for subjects in mimiciv_ext.cohort
# Requirements: pigz installed, .pgpass configured

set -euo pipefail

if [ -z "${DB:-}" ] || [ -z "${BASE_OUTPUT_DIR:-}" ]; then
  echo "Error: Required environment variables DB or BASE_OUTPUT_DIR are missing."
  echo "This script must be run via the Python wrapper or with variables explicitly exported."
  exit 1
fi

COHORT_TABLE="mimiciv_ext.cohort"

# Define Schemas to export (Output Folder Name -> Source DB Schema)
declare -A SCHEMAS
SCHEMAS["hosp"]="mimiciv_hosp"
SCHEMAS["icu"]="mimiciv_icu"

# --- Script Start ---

echo "Starting MIMIC-IV Cohort Data Export (Direct Filter)..."
echo "Target DB: $DB"
echo "Output Dir: $BASE_OUTPUT_DIR"
echo "Cohort Table: $COHORT_TABLE"

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
    # We exclude internal postgres tables or temporary tables if any
    tables=$(psql -d "$DB" -t -A -c "SELECT tablename FROM pg_tables WHERE schemaname='$source_schema' ORDER BY tablename")

    for table in $tables; do
        echo -n "  → Exporting $table... "

        # 1. Check if the table has a 'subject_id' column
        has_subject_id=$(psql -d "$DB" -t -A -c "SELECT count(*) FROM information_schema.columns WHERE table_schema='$source_schema' AND table_name='$table' AND column_name='subject_id'")

        # 2. Construct the COPY query
        if [ "$has_subject_id" -eq "1" ]; then
            # Case A: Table has subject_id
            # EXPORT LOGIC: Source Table INTERSECT Cohort Subjects
            SQL_QUERY="COPY (SELECT t.* FROM ${source_schema}.${table} t WHERE EXISTS (SELECT 1 FROM ${COHORT_TABLE} c WHERE c.subject_id = t.subject_id)) TO STDOUT CSV HEADER"
            echo -n "[Filtered: Cohort Only] "
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
echo "Export completed."
