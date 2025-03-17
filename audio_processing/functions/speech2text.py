# This script transcribes the audio and extracts the timestamps of each word.

import whisper

def transcribe_audio(audio_file_path, model_size="base"):
    """
    Transcribe audio file and extract word-level timestamps.
    
    Args:
        audio_file_path (str): Path to the audio file
        model_size (str): Whisper model size ("base", "small", "medium", "large")
    
    Returns:
        dict: Transcription results with timestamps
    """
    # Load the model
    model = whisper.load_model(model_size)

    # Transcribe with word timestamps enabled
    result = model.transcribe(audio_file_path, word_timestamps=True)
    
    word_starts = []
    word_ends = []
    word_texts = []

    # Extract segments with timestamps
    for segment in result["segments"]:
        # If you want word-level timestamps
        if "words" in segment:
            for word in segment["words"]:
                word_starts.append(word["start"])
                word_ends.append(word["end"])
                word_texts.append(word["word"])

    return word_starts, word_ends, word_texts

