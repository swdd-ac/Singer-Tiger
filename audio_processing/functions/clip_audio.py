from pydub import AudioSegment


def clip_audio(input_path, output_path, start_time_ms, end_time_ms):
    """
    Clips an audio file to a specified time range.
    
    Args:
        input_path (str): Path to the input audio file
        output_path (str): Path where the clipped audio will be saved
        start_time_ms (int): Start time in milliseconds
        end_time_ms (int): End time in milliseconds
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(input_path)
        
        # Extract the specified portion
        clipped_audio = audio[start_time_ms:end_time_ms]
        
        # Export the clipped audio
        clipped_audio.export(output_path, format=output_path.split('.')[-1])
        
        return True
    except Exception as e:
        print(f"Error clipping audio: {str(e)}")
        return False 