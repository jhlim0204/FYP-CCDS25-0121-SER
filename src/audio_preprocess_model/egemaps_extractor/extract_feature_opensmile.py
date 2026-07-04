import opensmile
import pandas as pd
import glob
import os


def extract_feature_opensmile(dataset, data_path, output_csv=None):
    # Initialize eGeMAPS (extended version)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPS,  # eGeMAPSv02
        feature_level=opensmile.FeatureLevel.Functionals
    )

    data_df = pd.read_csv(data_path)

    # Store results
    all_features = []

    # Loop through each audio file
    for file_path in data_df['path']:
        # Ensure the file exists and isn't a hidden system file
        if os.path.exists(file_path) and not os.path.basename(file_path).startswith('._'):
            # Extract audio features
            features = smile.process_file(file_path)
            
            # Add the 'path' column to the features so we can merge on it later
            features['path'] = file_path
            all_features.append(features)
            
    df = pd.concat(all_features)

    # Merge back into the original data_df using 'path' as the common key
    final_df = data_df.merge(df, on='path', how='inner')
    final_df.reset_index(drop=True, inplace=True)
    
    if output_csv is None:
        output_csv = f"../data/{dataset}_dataset/{dataset}_dataset_egemaps_features.csv"
    final_df.to_csv(output_csv, index=False)
    
    print("Features extracted and saved to egemaps_features.csv")

    return output_csv