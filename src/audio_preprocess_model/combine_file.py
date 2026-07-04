import pandas as pd

def combine_file_to_json(original_csv, gender_pred_csv, vad_pred_csv, output_path):
    # Load everything
    df_original = pd.read_csv(original_csv)
    df_gender = pd.read_csv(gender_pred_csv)
    df_vad = pd.read_csv(vad_pred_csv)

    final_df = df_original.merge(df_gender[['id', 'gender_pred', 'gender_probability']], on='id', how='left')
    final_df = final_df.merge(df_vad[['id', 'pred_valence', 'pred_arousal', 'pred_dominance']], on='id', how='left')

    for split in final_df['split'].unique():
        split_df = final_df[final_df['split'] == split]
        split_df.to_json(f"{output_path}/{split}.json", orient="records", indent=4)