import subprocess
import os
from pathlib import Path

def convert_m4a_to_wav(input_file: str, output_file: str = None) -> str:
    """
    Convert an M4A file to WAV format using ffmpeg.
    
    Args:
        input_file (str): Path to the input M4A file
        output_file (str, optional): Path for the output WAV file. If not provided,
                                   will use the same name as input with .wav extension
    
    Returns:
        str: Path to the converted WAV file
    
    Raises:
        FileNotFoundError: If the input file doesn't exist
        subprocess.CalledProcessError: If the ffmpeg conversion fails
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # If output file is not specified, create one with the same name but .wav extension
    if output_file is None:
        output_file = str(Path(input_file).with_suffix('.wav'))
    
    try:
        # Run ffmpeg command to convert the file
        subprocess.run([
            'ffmpeg',
            '-i', input_file,  # Input file
            '-acodec', 'pcm_s16le',  # Audio codec for WAV
            '-ar', '44100',  # Sample rate
            '-ac', '2',  # Number of audio channels (stereo)
            '-y',  # Overwrite output file if it exists
            output_file
        ], check=True, stderr=subprocess.PIPE)
        
        return output_file
    
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error converting file: {e.stderr.decode()}")

def convert_directory(input_dir: str, output_dir: str = None) -> list:
    """
    Convert all M4A files in a directory to WAV format.
    
    Args:
        input_dir (str): Path to the input directory containing M4A files
        output_dir (str, optional): Path to the output directory for WAV files
    
    Returns:
        list: List of paths to the converted WAV files
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # If output directory is not specified, use the input directory
    if output_dir is None:
        output_dir = input_dir
    else:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    converted_files = []
    
    # Process all M4A files in the input directory
    for file in Path(input_dir).glob('*.m4a'):
        output_file = os.path.join(output_dir, file.stem + '.wav')
        try:
            converted_file = convert_m4a_to_wav(str(file), output_file)
            converted_files.append(converted_file)
        except Exception as e:
            print(f"Error converting {file}: {str(e)}")
    
    return converted_files

if __name__ == '__main__':
    # Example usage
    try:
        # Convert a single file
        wav_file = convert_m4a_to_wav('example.m4a')
        print(f"Converted file: {wav_file}")
        
        # Convert all files in a directory
        converted_files = convert_directory('input_directory', 'output_directory')
        print(f"Converted {len(converted_files)} files")
        
    except Exception as e:
        print(f"Error: {str(e)}")
