#!/bin/bash
# Script to process job folders by running run_fcl.sh on wirecell_sim_output files


script_dir="/exp/sbnd/app/users/yuhw/dl-clustering/prep"
export FHICL_FILE_PATH=${script_dir}:${FHICL_FILE_PATH:-""}
echo ${FHICL_FILE_PATH} | tr ':' '\n'

# Check if input and output paths are provided
# if [ "$#" -lt 2 ]; then
#     echo "Usage: $0 <input_path> <output_path>"
#     exit 1
# fi
# input_path="$1"
# output_path="$2"
input_path="/exp/sbnd/app/users/yuhw/dl-clustering/artroot/20250618/"
output_path="/exp/sbnd/app/users/yuhw/dl-clustering/sample/20250618/"
job_batchid=77451011
start_job=0
end_job=9

# Ensure input path exists
if [ ! -d "$input_path" ]; then
    echo "Error: Input path '$input_path' does not exist"
    exit 1
fi

# Create output path if it doesn't exist
mkdir -p "$output_path"

# for job_folder in $(find "$input_path" -maxdepth 1 -type d -name "[0-9]*_[0-9]*"); do
    # Extract the job folder name from the full path
    # job_folder_name=$(basename "$job_folder")
for ((job_id=start_job; job_id<=end_job; job_id++)); do
    job_folder_name="${job_batchid}_${job_id}"
    job_folder="$input_path/$job_folder_name"
    # Check if the job folder exists
    if [ ! -d "$job_folder" ]; then
        echo "Warning: Job folder '$job_folder' does not exist, skipping..."
        continue
    fi

    echo "Processing job folder: $job_folder_name"
    
    # Create corresponding output folder
    out_folder="$output_path/$job_folder_name"
    mkdir -p "$out_folder"
    
    # Find wirecell_sim_output*.root files and process them
    for input_file in $(find "$job_folder" -name "wirecell_sim_output*.root"); do
        echo "Processing input file: $(basename "$input_file")"
        
        # Change to the output directory before running the script
        cd "$out_folder"
        
        # Run the run_fcl.sh script with nevents=-1
        echo "Running: $script_dir/run_fcl.sh $input_file -1"
        "$script_dir/run_fcl.sh" "$input_file" -1
    done
done

echo "All job folders processed successfully!"
