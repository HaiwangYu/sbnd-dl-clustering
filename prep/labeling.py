import os
import sys
import numpy as np
import json
import argparse
from labeler import get_isnu_labels

def process_file_pair(tru_file, rec_file, out_file, max_distance=5):
    """
    Process a pair of truth and reconstruction files:
    1. Extract labels from truth data for points in the rec file
    2. Add these labels to the rec file data
    3. Save the labeled data to the output file
    
    Parameters:
    -----------
    tru_file : str
        Path to the truth data JSON file
    rec_file : str
        Path to the reconstruction NPZ file
    out_file : str
        Path to save the output NPZ file with labels
    max_distance : float, optional
        Maximum allowed distance for matching, default is 5
    """
    # Check if files exist
    if not os.path.exists(tru_file):
        raise FileNotFoundError(f"Truth file {tru_file} not found.")
    if not os.path.exists(rec_file):
        raise FileNotFoundError(f"Reconstruction file {rec_file} not found.")
    
    # Get labels using the function from labeler.py
    labels = get_isnu_labels(tru_file, rec_file, max_distance)
    
    # Load reconstruction data
    rec_data = np.load(rec_file)
    
    # Create a dictionary for the output data
    output_data = {}
    
    # Copy all fields from the original rec_file
    for key in rec_data.keys():
        output_data[key] = rec_data[key]
    
    # Add the is_nu labels
    output_data['is_nu'] = labels
    
    # Save the output data
    np.savez(out_file, **output_data)
    
    return len(labels)

def process_files(tru_file_prefix, rec_file_prefix, rec_lab_file_prefix, entries, max_distance=5):
    """
    Process multiple pairs of files based on entry numbers
    
    Parameters:
    -----------
    tru_file_prefix : str
        Prefix for truth files
    rec_file_prefix : str
        Prefix for reconstruction files
    rec_lab_file_prefix : str
        Prefix for output files with labels
    entries : list
        List of entry numbers to process
    max_distance : float, optional
        Maximum allowed distance for matching, default is 5
    """
    results = {}
    
    for entry in entries:
        tru_file = f"{tru_file_prefix}-{entry}.json"
        rec_file = f"{rec_file_prefix}-{entry}.npz"
        out_file = f"{rec_lab_file_prefix}-{entry}.npz"
        
        print(f"Processing files: {tru_file} and {rec_file}")
        try:
            n_labels = process_file_pair(tru_file, rec_file, out_file, max_distance)
            print(f"Successfully processed files and saved to {out_file} with {n_labels} labels")
            results[entry] = {"status": "success", "labels": n_labels}
        except Exception as e:
            print(f"Error processing entry {entry}: {str(e)}")
            results[entry] = {"status": "failed", "error": str(e)}
    
    return results

def main():
    """Main function to parse arguments and run the processing"""
    parser = argparse.ArgumentParser(description='Add is_nu labels to reconstruction files based on truth data.')
    parser.add_argument('--tru-prefix', required=True, help='Prefix for truth files (tru_prefix-entry.json)')
    parser.add_argument('--rec-prefix', required=True, help='Prefix for reconstruction files (rec_prefix-entry.npz)')
    parser.add_argument('--out-prefix', required=True, help='Prefix for output files (out_prefix-entry.npz)')
    parser.add_argument('--entries', required=True, help='Entry numbers to process (comma-separated or range: start-end)')
    parser.add_argument('--max-distance', type=float, default=5, help='Maximum distance for matching, default is 5')
    
    args = parser.parse_args()
    
    # Parse entries
    if '-' in args.entries:
        start, end = map(int, args.entries.split('-'))
        entries = range(start, end + 1)
    else:
        entries = [int(e) for e in args.entries.split(',')]
    
    # Process files
    results = process_files(args.tru_prefix, args.rec_prefix, args.out_prefix, entries, args.max_distance)
    
    # Print summary
    print("\nProcessing Summary:")
    successful = sum(1 for r in results.values() if r["status"] == "success")
    failed = sum(1 for r in results.values() if r["status"] == "failed")
    print(f"Total entries: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
