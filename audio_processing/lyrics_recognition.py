import whisper
import torch
import os
import json

def transcribe_audio(audio_path, output_dir="./analysis/whisper", model_size="medium", verbose=False):
    """
    Transcribe audio file to text with word-level timestamps
    
    Args:
        audio_path (str): Path to the audio file
        output_dir (str): Directory to save the transcript (default: ./analysis/whisper)
        model_size (str): Whisper model size to use (default: medium)
        verbose (bool): Whether to print progress (default: False)
    
    Returns:
        str: Path to the saved transcript JSON file
    """
    # Load the Whisper model
    model = whisper.load_model(model_size)
    
    # Transcribe the audio file with word timestamps
    result = model.transcribe(audio_path, verbose=verbose, word_timestamps=True)
    
    # Prepare output data
    output_data = []
    for segment in result["segments"]:
        for word in segment["words"]:
            word_data = {
                "start_time": format_timestamp(word["start"]),
                "end_time": format_timestamp(word["end"]),
                "text": word["word"].strip()
            }
            output_data.append(word_data)
            if verbose:
                print(f"[{word_data['start_time']} -> {word_data['end_time']}] {word_data['text']}")
    
    # Create output filename and directory
    base_name = os.path.basename(os.path.splitext(audio_path)[0])
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{base_name}_transcript.json")
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    if verbose:
        print(f"\nTranscript saved to: {output_file}")
    
    return output_file

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS.mmm format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_decimal = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds_decimal:06.3f}"

# This allows the file to be imported as a module or run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe audio file to text with word-level timestamps")
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument("--output_dir", default="./analysis/whisper", help="Directory to save the transcript")
    parser.add_argument("--model_size", default="medium", help="Whisper model size to use")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    
    args = parser.parse_args()
    
    transcribe_audio(args.audio_path, args.output_dir, args.model_size, args.verbose) 