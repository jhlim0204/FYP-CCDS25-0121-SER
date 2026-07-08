import argparse
import json
import logging
import os
import sys
from pathlib import Path
import pandas as pd

from audio_preprocess_model import (
    run_gender_classifier_inference,
    inference_audeering,
    combine_file_to_json,
    extract_feature_opensmile,
    process_custom_audio_feature,
    run_asr
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("preprocess_inference")

def prepare_manifest(input_path: str, temp_dir: Path) -> Path:
    """Creates a standardized manifest CSV whether given a single .wav or an existing .csv."""
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.suffix.lower() == ".wav":
        manifest_path = temp_dir / "input_manifest.csv"
        df = pd.DataFrame([{
            "id": path.stem.replace(" ", "_"),
            "path": str(path),
            # "audio_filepath": str(path),
            "split": "test",
            "utterance": ""
        }])
        df.to_csv(manifest_path, index=False)
        logger.info(f"Generated temporary 1-row manifest for WAV file: {manifest_path}")
        return manifest_path
    elif path.suffix.lower() == ".csv":
        logger.info(f"Using provided CSV manifest directly: {path}")
        return path
    else:
        raise ValueError("Input must be either a .wav audio file or a .csv manifest file.")

# def generate_asr_json(manifest_csv: Path, json_out_path: Path) -> None:
#     """Converts a CSV manifest into the JSON format required by run_asr."""
#     df = pd.read_csv(manifest_csv)
#     # Match column conventions cleanly
#     path_col = "path" if "path" in df.columns else ("audio_filepath" if "audio_filepath" in df.columns else df.columns[1])
#     id_col = "id" if "id" in df.columns else df.columns[0]
    
#     payload = []
#     for _, row in df.iterrows():
#         payload.append({
#             "id": str(row[id_col]),
#             "path": str(row[path_col]),
#             "utterance": str(row.get("utterance", ""))
#         })
#     json_out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
#     logger.info(f"Formatted ASR input manifest: {json_out_path}")

def main():
    parser = argparse.ArgumentParser(description="Standalone Audio & Feature Preprocessing for SER Inference")
    parser.add_argument("--input_path", required=True, help="Path to single .wav file or .csv manifest")
    parser.add_argument("--dataset", required=True, choices=["iemocap", "msp"], help="Target dataset format")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to downloaded checkpoint directory")
    parser.add_argument("--temp_dir", required=True, help="Directory to store intermediate artifacts")
    parser.add_argument("--gender_extractor_path", default="facebook/wav2vec2-base")
    parser.add_argument("--gender_classifier_path", default=None)
    parser.add_argument("--vad_extractor_path", default="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
    parser.add_argument("--use_default_vad_model", action="store_true", help="Use default VAD model if no specific path is provided")
    parser.add_argument("--vad_model_path", default=None, help="Optional specific VAD model path; if not provided, uses downloaded checkpoint, or trained checkpoint if --use_default_vad_model is set")
    parser.add_argument("--asr_model_id", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--use_pred_gender", action="store_true", help="Use predicted gender for feature normalization")
    parser.add_argument("--use_asr_pred", action="store_true", help="Use ASR predictions for feature normalization")
    parser.add_argument("--with_history", required=True, choices=["True", "False"], help="Whether to include historical context")
    
    args = parser.parse_args()
    temp_dir = Path(args.temp_dir).resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 0 Initialise with_history flag
    if args.with_history == 'True':
        args.with_history = True
    elif args.with_history == 'False':
        args.with_history = False
        
    # 1. Prepare Manifest
    manifest_csv = prepare_manifest(args.input_path, temp_dir)

    # 2. Gender Classifier Inference
    logger.info("--- Step 1/5: Running Gender Classifier ---")
    if args.gender_classifier_path is None:
        gender_ckpt = os.path.join(args.checkpoint_dir, "Gender_Classifier_checkpoint")
    # if not os.path.exists(gender_ckpt):
    #     gender_ckpt = args.gender_extractor_path # Fallback to base model if subfolder missing
    gender_out_csv = temp_dir / "gender_predictions.csv"
    
    run_gender_classifier_inference(
        data_path=str(manifest_csv),
        extractor_path=args.gender_extractor_path,
        model_path=gender_ckpt,
        output_csv=str(gender_out_csv)
    )

    # 3. VAD Regressor Inference
    logger.info("--- Step 2/5: Running VAD Regressor ---")
    vad_out_path = temp_dir / f"{args.dataset}_vad_predictions.csv"
    if args.use_default_vad_model:
        vad_ckpt = None
    elif args.vad_model_path is None:
        vad_ckpt = os.path.join(args.checkpoint_dir, "VAD_Regressor_checkpoint")
    
    vad_out_csv = inference_audeering(
        EXTRACTOR_PATH=args.vad_extractor_path,
        input_data=str(manifest_csv),
        output_path=str(vad_out_path),
        dataset=args.dataset,
        Model_PATH=vad_ckpt
    )

    # 4. Combine and split into test.json (and train.json if applicable)
    logger.info("--- Step 3/5: Extracting eGemaps Features and Combining files---")
    feature_output_csv=str(temp_dir / "egemaps_features.csv")
    combined_csv = extract_feature_opensmile(dataset=args.dataset, data_path=str(manifest_csv), output_csv=feature_output_csv)
    combine_file_to_json(
        original_csv=str(feature_output_csv),
        gender_pred_csv=str(gender_out_csv),
        vad_pred_csv=str(vad_out_csv),
        output_path=str(temp_dir)
    )

    # 5. ASR Speech-to-Text
    logger.info("--- Step 4/5: Running Whisper Speech-to-Text ---")
    asr_json = temp_dir / "test.json"
    
    run_asr(
        input_json=str(asr_json),
        output_json=str(asr_json),
        model_id=args.asr_model_id,
        calc_wer=False,
        normalize_wer=False
    )
    
    # 6. Feature Extraction
    logger.info("--- Step 5/5: Processing eGeMaps Features ---")
    process_custom_audio_feature(
        dataset=args.dataset,
        output_path=str(temp_dir),
        use_pred_gender=True,
        use_asr_pred=True,
        with_history=args.with_history
    )

    logger.info("Audio Preprocessing Completed Successfully!")

if __name__ == "__main__":
    main()
