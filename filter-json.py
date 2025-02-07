import json

def filter_json_by_cluster_id(json_input_path, json_output_path, target_cluster_id=0):
    """
    Reads a JSON file, filters entries by cluster_id == target_cluster_id,
    and writes the filtered data to a new JSON file.

    Expects a JSON structure like:
    {
      "x": [0, 1, ...],
      "y": [2, 3, ...],
      "cluster_id": [0, 1, ...]
      // possibly other arrays of the same length
    }
    """
    # 1. Read the JSON file
    with open(json_input_path, 'r') as f:
        data = json.load(f)
    
    # 2. Extract the cluster_id array
    cluster_ids = data["cluster_id"]

    # 3. Identify which entries have cluster_id == target_cluster_id
    indices_to_keep = [i for i, cid in enumerate(cluster_ids) if cid == target_cluster_id]

    # 4. Create a new dictionary to store the filtered data
    filtered_data = {}
    
    # We assume all keys in the JSON correspond to lists of the same length.
    # Filter each list based on indices_to_keep.
    for key, values in data.items():
        print(f"Filtering {key}: {len(values)} -> {len(indices_to_keep)}")
        # Ensure the value is a list (or similar) before indexing
        # If there's a single value or something else, handle accordingly.
        if isinstance(values, list):
            filtered_data[key] = [values[i] for i in indices_to_keep]
        else:
            # If it's not a list, decide if you want to copy it as is or ignore
            filtered_data[key] = values

    # 5. Write the filtered data to a new JSON file
    with open(json_output_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)

# Example usage:
if __name__ == "__main__":
    input_path = "bee/data/0/0-truthDepo.json"
    output_path = "bee/data/0/0-truthDepo-filter.json"
    filter_json_by_cluster_id(input_path, output_path, target_cluster_id=0)
    print(f"Filtered JSON saved to {output_path}")
