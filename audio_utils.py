"""
Audio utility functions for file validation and processing
"""
import streamlit as st
import tempfile
import os
from typing import Dict, Any

def validate_audio_file(audio_file) -> bool:
    """
    Validate if uploaded file is a supported audio format
    
    Args:
        audio_file: Streamlit uploaded file object
        
    Returns:
        bool: True if valid audio file, False otherwise
    """
    if audio_file is None:
        return False
    
    # Check file extension
    valid_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
    file_extension = os.path.splitext(audio_file.name)[1].lower()
    
    if file_extension not in valid_extensions:
        return False
    
    # Check file size (limit to 100MB)
    max_size = 100 * 1024 * 1024  # 100MB in bytes
    if hasattr(audio_file, 'size') and audio_file.size > max_size:
        st.error("File size exceeds 100MB limit")
        return False
    
    return True

def get_audio_info(audio_file) -> Dict[str, Any]:
    """
    Extract basic information from audio file
    
    Args:
        audio_file: Streamlit uploaded file object
        
    Returns:
        Dict with audio information
    """
    try:
        # Mock audio information since we can't actually analyze the file
        # In a real implementation, this would use librosa, pydub, or similar
        file_extension = os.path.splitext(audio_file.name)[1].lower()
        file_size = getattr(audio_file, 'size', 0)
        
        # Mock duration based on file size (rough estimate)
        estimated_duration = max(1.0, min(file_size / (44100 * 2 * 2), 300.0))  # Estimate max 5 minutes
        
        return {
            'filename': audio_file.name,
            'format': file_extension[1:].upper(),
            'duration': estimated_duration,
            'size_bytes': file_size,
            'size_mb': round(file_size / (1024 * 1024), 2)
        }
    except Exception as e:
        st.error(f"Error analyzing audio file: {str(e)}")
        return {
            'filename': audio_file.name if audio_file else 'Unknown',
            'format': 'Unknown',
            'duration': 0.0,
            'size_bytes': 0,
            'size_mb': 0.0
        }

def save_uploaded_file(uploaded_file, suffix='.wav'):
    """
    Save uploaded file to temporary location
    
    Args:
        uploaded_file: Streamlit uploaded file object
        suffix: File suffix to use
        
    Returns:
        str: Path to saved temporary file
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration (e.g., "2:34")
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"

def get_supported_formats() -> list:
    """
    Get list of supported audio formats
    
    Returns:
        list: List of supported file extensions
    """
    return ['wav', 'mp3', 'flac', 'm4a', 'ogg', 'aac']

def estimate_processing_time(duration: float, quality: str = "Balanced") -> float:
    """
    Estimate processing time based on audio duration and quality setting
    
    Args:
        duration: Audio duration in seconds
        quality: Processing quality setting
        
    Returns:
        float: Estimated processing time in seconds
    """
    base_multiplier = {
        "Fast": 0.5,
        "Balanced": 1.0,
        "High Quality": 2.0
    }
    
    multiplier = base_multiplier.get(quality, 1.0)
    # Base processing time is roughly 0.3x the duration of the audio
    estimated_time = duration * 0.3 * multiplier
    
    # Add some variance and minimum processing time
    return max(2.0, min(estimated_time, 60.0))  # Between 2 seconds and 1 minute
