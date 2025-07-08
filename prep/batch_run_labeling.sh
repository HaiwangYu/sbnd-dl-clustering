#!/bin/bash
# Script to process job folders by running run_fcl.sh on wirecell_sim_output files


script_dir="/exp/sbnd/app/users/yuhw/dl-clustering/prep"

export PYTHONPATH=${script_dir}:${PYTHONPATH:-""}
echo ${PYTHONPATH} | tr ':' '\n'

# Check if input and output paths are provided
# if [ "$#" -lt 2 ]; then
#     echo "Usage: $0 <input_path> <output_path>"
#     exit 1
# fi
# input_path="$1"
# output_path="$2"
input_path="/exp/sbnd/app/users/yuhw/dl-clustering/sample/20250618/"
output_path="/exp/sbnd/app/users/yuhw/dl-clustering/sample/20250618/"

# Ensure input path exists
if [ ! -d "$input_path" ]; then
    echo "Error: Input path '$input_path' does not exist"
    exit 1
fi

# Create output path if it doesn't exist
mkdir -p "$output_path"

# Find job folders matching the pattern <jobid>_<subjobid>
for job_folder in $(find "$input_path" -maxdepth 1 -type d -name "[0-9]*_[0-9]*"); do
    # Extract the job folder name from the full path
    job_folder_name=$(basename "$job_folder")
    
    echo "Processing job folder: $job_folder_name"
    
    # Create corresponding output folder
    out_folder="$output_path/$job_folder_name"
    mkdir -p "$out_folder"

    # Find all files matching the pattern rec-apa0-*.npz
    echo "Finding maximum event number in $job_folder ..."
    max_event=-1
    for npz_file in $(find "$job_folder" -name "rec-apa0-*.npz" -type f); do
        # Extract event number from filename
        filename=$(basename "$npz_file")
        echo "Processing file: $filename"
        event_num=$(echo "$filename" | sed -E 's/rec-apa0-([0-9]+)\.npz/\1/')
        # Check if it's a valid number and compare with current max
        if [[ "$event_num" =~ ^[0-9]+$ ]]; then
            if [ "$event_num" -gt "$max_event" ]; then
                max_event="$event_num"
            fi
        fi
    done
    echo "Maximum event number in $job_folder_name: $max_event"

    # Run the run_fcl.sh script with nevents=-1
    if [ "$max_event" -eq -1 ]; then
        echo "No rec-apa0-*.npz files found in $job_folder. Skipping labeling."
        continue
    fi

    cd "$out_folder"
    ln -sf "$script_dir/labeling.py" .
    echo "Running: $script_dir/run_labeling.sh 0 $max_event"
    "$script_dir/run_labeling.sh" 0 $max_event
done

echo "All job folders processed successfully!"
