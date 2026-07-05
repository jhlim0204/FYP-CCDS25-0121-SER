import pandas as pd
import os
import numpy as np
from configs.dataset_config import FEATURE_RENAMING_MAPS, DATASET_PROMPT_GROUPS, EXTRACTED_FEATURE_SET
import json

def extract_thresholds_and_stats(df, dataset, num_classes, compute_manual=True):
    if not compute_manual:
        THRESHOLDS_PATH = f"configs/precomputed/{dataset}_thresholds.json"
        STATS_PATH = f"configs/precomputed/{dataset}_stats.json"

        if not os.path.exists(THRESHOLDS_PATH) or not os.path.exists(STATS_PATH):
            raise FileNotFoundError(
                f"Precomputed files missing for dataset '{dataset}'. "
                f"Expected paths:\n  - {THRESHOLDS_PATH}\n  - {STATS_PATH}\n"
                f"Please run with `compute_manual=True` first to generate them."
            )

        try:
            # Load them dynamically if they exist
            if os.path.exists(THRESHOLDS_PATH):
                with open(THRESHOLDS_PATH, "r") as f:
                    thresholds = json.load(f)

            if os.path.exists(STATS_PATH):
                with open(STATS_PATH, "r") as f:
                    stats = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse precomputed JSON files for '{dataset}'. "
                f"The files might be corrupted or empty. Error detail: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while loading precomputed data: {e}"
            ) from e

        return thresholds, stats

    
    features = EXTRACTED_FEATURE_SET[dataset]
    thresholds = {}
    stats = {}
    
    # Calculate overall thresholds and stats
    overall_thresholds = {}
    overall_stats = {}
    for feature in features:
        feature_data = df[feature].replace(0, np.nan).dropna()
        if num_classes == 5:
            q1, q2, q3, q4 = feature_data.quantile([0.1, 0.25, 0.75, 0.9])
            overall_thresholds[feature] = {'very_low': q1, 'low': q2, 'medium': q3, 'high': q4}
        overall_stats[feature] = {'mean': feature_data.mean(), 'std': feature_data.std()}
    
    thresholds['overall'] = overall_thresholds
    stats['overall'] = overall_stats
    
    # Calculate gender-specific thresholds, stats, and plot
    gender_list = df['gender'].unique().tolist()
    for gender in gender_list:
        gender_df = df[df['gender'] == gender]
        gender_thresholds = {}
        gender_stats = {}
        for feature in features:
            feature_data = gender_df[feature].replace(0, np.nan).dropna()
            if len(feature_data) > 0:
                if num_classes == 5:
                    q1, q2, q3, q4 = feature_data.quantile([0.1, 0.25, 0.75, 0.9])
                    gender_thresholds[feature] = {'very_low': q1, 'low': q2, 'medium': q3, 'high': q4}
                gender_stats[feature] = {'mean': feature_data.mean(), 'std': feature_data.std()}
            else:
                gender_thresholds[feature] = overall_thresholds[feature]
                gender_stats[feature] = overall_stats[feature]
        thresholds[gender] = gender_thresholds
        stats[gender] = gender_stats
    
    return thresholds, stats

def categorize(value, thresholds, num_classes):
    if pd.isna(value) or value == 0:
        return 'none'

    elif num_classes == 5:
        if value <= thresholds['very_low']:
            return 'Very low'
        elif value <= thresholds['low']:
            return 'Low'
        elif value <= thresholds['medium']:
            return 'Medium'
        elif value <= thresholds['high']:
            return 'High'
        else:
            return 'Very high'

def standardize_and_process_df(df, thresholds, stats, num_classes, use_pred_gender=False):
    """
    Standardizes and processes the DataFrame with the given thresholds and stats.
    """
    features = list(thresholds['overall'].keys())

    def _get_gender_key(row):
        if use_pred_gender:
            gender_prob = row.get('gender_probability', np.nan)
            if not pd.isna(gender_prob) and 0.2 <= gender_prob <= 0.8:
                return 'overall'
            
            key = row.get('gender_pred', np.nan)
        else:
            key = row.get('gender', np.nan)
            
        if pd.isna(key) or key not in stats:
            return 'overall'
        return key
    
    # Standardize features
    for feature in features:
        df[f'{feature}_standardized'] = df.apply(lambda row: 
            (row[feature] - stats.get(_get_gender_key(row), stats['overall'])[feature]['mean']) / 
            stats.get(_get_gender_key(row), stats['overall'])[feature]['std']
        if not pd.isna(row[feature]) and row[feature] != 0 else np.nan, axis=1)
    
    # Categorize original features
    for feature in features:
        df[f'{feature}_category'] = df.apply(lambda row: categorize(
            row[feature], 
            thresholds.get(_get_gender_key(row), thresholds['overall'])[feature],
            num_classes
        ), axis=1)
    
    return df

# For IEMOCAP
def add_conversation_history(df, window_size=8, use_pred_gender=False, use_asr_pred=False):
    """
    Creates a 'history_str' for IEMOCAP using strict logic from data_process.py.
    
    Logic:
      - Window: Current utterance + previous 'window_size' utterances.
      - Format: Tab-separated (\t), matching the original script.
      - Speakers: Uses df['gender'] (e.g., 'M'/'F'). 
        (Note: Original script mapped M->0, F->1. If you prefer that, 
         you can map the column before running this.)
    """
    # 1. Sort to ensure temporal order
    # Assuming 'segment_id' or 'turn_index' exists to order utterances within a dialogue
    df = df.sort_values(by=['video_id', 'Order_Index']) 
    
    # Initialize column
    df['history_context'] = ""
    
    # 2. Iterate through each distinct conversation
    for conversation_id, group in df.groupby('video_id', sort=False):
        if use_pred_gender:
            speakers = group['gender_pred'].astype(str).tolist()
        else:
            speakers = group['gender'].astype(str).tolist()

        # 2. Determine which text column to pull from the current group
        if use_asr_pred:
            texts = group['hypothesis'].astype(str).tolist()
        else:
            texts = group['text'].astype(str).tolist()
        
        indices = group.index.tolist()
        pitches = group['Average Pitch_category'].tolist()
        variations = group['Pitch Stability (StdDev)_category'].tolist()
 
        for i, row_idx in enumerate(indices):
            start_pos = max(0, i - window_size)
            end_pos = i + 1
            
            # Slice the lists
            w_texts = texts[start_pos:end_pos]
            w_speakers = speakers[start_pos:end_pos]
            w_pitches = pitches[start_pos:end_pos]
            w_variations = variations[start_pos:end_pos]

            lines = []
            window_len = len(w_texts)

            # Iterate through the current window slice
            for k in range(window_len):
                s = w_speakers[k]
                u = w_texts[k]
                p = w_pitches[k]
                v = w_variations[k]

                # Base string
                utterance_str = f'Speaker_{s}:"{u}"'

                # Add features only to the last 3 items
                # "k" is the index within this specific window (0 to window_len-1).
                # If window has 5 items, indices are 0,1,2,3,4. We want 2,3,4.
                if k >= window_len - 3:
                    utterance_str += f' ({p} pitch with {v} variation)'
                
                lines.append(utterance_str)

            df.at[row_idx, 'history_context'] = "\t " + "\t ".join(lines)

    return df

# For MSP
def add_one_line_convo(processed_df, use_pred_gender=False, use_asr_pred=False):
    # temp_content_str = 'The following noted between \'### ###\' is a single isolated utterance with its speech features attached. ### '
    # temp_content_str += f"\t Speaker_{row['gender']}: {row['transcription']}"
    # temp_content_str += f" ({row['Average Pitch_category']} pitch with {row['Pitch Stability (StdDev)_category']} variation)  ### \n"

    # return temp_content_str
    prefix = "The following noted between \'### ###\' is a single isolated utterance with its speech features attached. ### "
    
    if use_pred_gender:
        gender_str = processed_df['gender_pred'].astype(str)
    
    else:
        gender_str = processed_df['gender'].astype(str)

    if use_asr_pred:
        text_str = processed_df['hypothesis']
    
    else:
        text_str = processed_df['text']
        
    # Vectorized string concatenation
    processed_df['history_context'] = (
        prefix + 
        "\t Speaker_" + gender_str + ": " + 
        text_str +
        " (" + processed_df['Average Pitch_category'] + " pitch with " + 
        processed_df['Pitch Stability (StdDev)_category'] + " variation)  ### \n"
    )
    
    return processed_df

def add_conversation_history_custom_dataset(
    df, 
    id_col='id',
    window_size=8, 
    use_pred_gender=False, 
    use_asr_pred=False
):
    """
    Creates a 'history_context' for custom datasets based on an ID column formatted as 'videoID_uttrID'.
    
    Parameters:
      df (pd.DataFrame): Input dataframe.
      id_col (str): Column containing IDs formatted as 'videoID_uttrID' (e.g., 'conv1_0', 'conv1_1').
      window_size (int): Max previous utterances to include in context.
      use_pred_gender (bool): Whether to use 'gender_pred' instead of 'gender'.
      use_asr_pred (bool): Whether to use 'hypothesis' instead of 'text'.
    """
    
    # 1. Parse video_id and uttr_id using the LAST occurrence of '_'
    # expand=True returns a DataFrame with 2 columns: [video_id, uttr_id]
    extracted_ids = df[id_col].astype(str).str.rsplit('_', n=1, expand=True)
    
    df['_temp_video_id'] = extracted_ids[0]
    
    # Safely convert uttr_id to integers for correct numerical sorting (e.g., '10' after '9', not after '1')
    # If conversion fails (e.g., alphanumeric IDs like 'uttr1'), fallback to string sorting
    try:
        df['_temp_uttr_order'] = pd.to_numeric(extracted_ids[1])
    except (ValueError, TypeError):
        df['_temp_uttr_order'] = extracted_ids[1]

    # 2. Sort to ensure strict temporal order within each dialogue
    df = df.sort_values(by=['_temp_video_id', '_temp_uttr_order'])
    
    # Initialize output column
    df['history_context'] = ""
    
    # 3. Iterate through each distinct conversation
    for conversation_id, group in df.groupby('_temp_video_id', sort=False):
        # Resolve speakers
        if use_pred_gender and 'gender_pred' in group:
            speakers = group['gender_pred'].astype(str).tolist()
        else:
            speakers = group['gender'].astype(str).tolist()

        # Resolve texts
        if use_asr_pred and 'hypothesis' in group:
            texts = group['hypothesis'].astype(str).tolist()
        else:
            texts = group['text'].astype(str).tolist()
        
        indices = group.index.tolist()
        
        # Safe feature extraction (defaults to 'N/A' if columns are missing in custom dataset)
        pitches = group['Average Pitch_category'].tolist() if 'Average Pitch_category' in group else ['N/A'] * len(group)
        variations = group['Pitch Stability (StdDev)_category'].tolist() if 'Pitch Stability (StdDev)_category' in group else ['N/A'] * len(group)

        for i, row_idx in enumerate(indices):
            start_pos = max(0, i - window_size)
            end_pos = i + 1
            
            # Slice the rolling window
            w_texts = texts[start_pos:end_pos]
            w_speakers = speakers[start_pos:end_pos]
            w_pitches = pitches[start_pos:end_pos]
            w_variations = variations[start_pos:end_pos]

            lines = []
            window_len = len(w_texts)

            # Build the contextual history strings
            for k in range(window_len):
                s = w_speakers[k]
                u = w_texts[k]
                p = w_pitches[k]
                v = w_variations[k]

                # Base utterance format
                utterance_str = f'Speaker_{s}:"{u}"'

                # Append acoustic metadata only to the last 3 items in the window
                if k >= window_len - 3:
                    utterance_str += f' ({p} pitch with {v} variation)'
                
                lines.append(utterance_str)

            # Assign formatted tab-separated string back to the dataframe
            df.at[row_idx, 'history_context'] = "\t " + "\t ".join(lines)

    # 4. Clean up temporary sorting columns
    df = df.drop(columns=['_temp_video_id', '_temp_uttr_order'])

    return df

def prepare_and_save_json(df, dataset, output_path, use_asr_pred=False):
    print(f"Processing {len(df)} rows...")
    final_columns = {} # Dict to map {Old_Name : New_Name}
    
    group_order = DATASET_PROMPT_GROUPS.get(dataset)
    
    # 2a. Add Metadata columns
    final_columns['id'] = 'id'
    if use_asr_pred:
        final_columns['hypothesis'] = 'utterance'
    else:
        final_columns['text'] = 'utterance'
    final_columns['emotion'] = 'output'
    final_columns['path'] = 'path'
    final_columns['history_context'] = 'history_context'
    
    if 'pred_valence' in df.columns: 
        df.drop(columns=['valence'], inplace=True, errors='ignore')
        final_columns['pred_valence'] = 'valence'
        
    if 'pred_arousal' in df.columns: 
        df.drop(columns=['arousal'], inplace=True, errors='ignore')
        final_columns['pred_arousal'] = 'arousal'
        
    if 'pred_dominance' in df.columns:
        df.drop(columns=['dominance'], inplace=True, errors='ignore')
        final_columns['pred_dominance'] = 'dominance'
    
    # 2b. Add Acoustic columns (and ensure they exist)
    for group, features in group_order.items():
        for feature in features:
            # Check if your DF has "Average Pitch" OR "Average Pitch_category"
            if f"{feature}_category" in df.columns:
                final_columns[f"{feature}_category"] = f"{feature}_category"
            elif feature in df.columns:
                final_columns[feature] = f"{feature}_category" # Rename it!
            else:
                # If missing, create a placeholder so code doesn't crash
                print(f"Warning: Missing {feature}, filling with 'N/A'")
                df[f"{feature}_category"] = "N/A"
                final_columns[f"{feature}_category"] = f"{feature}_category"   
    
    iemocap_to_target = {
        'hap': 'happy',
        'happy': 'happy',

        'sad': 'sad',

        'neu': 'neutral',
        'neutral': 'neutral',

        'ang': 'angry',
        'anger': 'angry',

        'exc': 'excited',
        'excited': 'excited',

        'fru': 'frustrated',
        'frustrated': 'frustrated',
    }
    if 'emotion' in df.columns:
        df['emotion'] = df['emotion'].map(iemocap_to_target)
        
    # export_df = df[list(final_columns.keys())].rename(columns=final_columns)
    # Filter keys to only those present in df
    existing_keys = [col for col in final_columns.keys() if col in df.columns]

    # Slice using existing keys, then rename using the original map
    export_df = df[existing_keys].rename(columns=final_columns)
    
    export_df.to_json(
        output_path, 
        orient='records', 
        indent=4, 
        force_ascii=False
    )
    
    print(f"Saved to {output_path}")

def process_audio_feature(dataset, output_path, use_pred_gender=False, use_asr_pred=False):    
    train_json = os.path.join(output_path, 'train.json')
    test_json = os.path.join(output_path, 'test.json')
    
    # Load the dataset
    train_df = pd.read_json(train_json)
    train_df.rename(columns=FEATURE_RENAMING_MAPS[dataset], inplace=True)
    
    test_df = pd.read_json(test_json)
    test_df.rename(columns=FEATURE_RENAMING_MAPS[dataset], inplace=True)
    
    num_classes = 5  
    
    # Extract thresholds and stats based on the training data
    thresholds, stats = extract_thresholds_and_stats(train_df, dataset, num_classes)
    
    # Process the entire dataset
    df = pd.concat([train_df, test_df])
    processed_df = standardize_and_process_df(df, thresholds, stats, num_classes)
    
    if dataset == "msp":
        processed_df = add_one_line_convo(processed_df, use_pred_gender, use_asr_pred)

    elif dataset == "iemocap":
        processed_df = add_conversation_history(processed_df, 8, use_pred_gender, use_asr_pred)

    # Create a new DataFrame with only the desired columns
    df_train = processed_df[processed_df['split']=='train'].copy()
    df_test = processed_df[processed_df['split']=='test'].copy()
    
    prepare_and_save_json(df_train, dataset, train_json)
    prepare_and_save_json(df_test, dataset, test_json)

def process_custom_audio_feature(dataset, output_path, use_pred_gender=True, use_asr_pred=True, with_history=True):    
    test_json = os.path.join(output_path, 'test.json')
    
    test_df = pd.read_json(test_json)
    test_df.rename(columns=FEATURE_RENAMING_MAPS[dataset], inplace=True)
    
    num_classes = 5  
    
    # Extract thresholds and stats based on the training data
    thresholds, stats = extract_thresholds_and_stats(test_df, dataset, num_classes, compute_manual=False)
    
    # Process the entire dataset
    processed_df = standardize_and_process_df(test_df, thresholds, stats, num_classes)
    
    if with_history:
        processed_df = add_conversation_history_custom_dataset(
            processed_df, 
            id_col='id', 
            window_size=8, 
            use_pred_gender=True, 
            use_asr_pred=True)
    else:
        processed_df = add_one_line_convo(processed_df, use_pred_gender=True, use_asr_pred=True)

    # Create a new DataFrame with only the desired columns
    prepare_and_save_json(processed_df, dataset, test_json, use_asr_pred=True)

