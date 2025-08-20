"""
Mock AI processing classes for voice conversion and vocal separation
"""
import time
import numpy as np
import io
from typing import Tuple

class MockVoiceConverter:
    """Mock voice conversion processor"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_loaded = False
    
    def load_model(self):
        """Simulate model loading"""
        if not self.is_loaded:
            time.sleep(1)  # Simulate loading time
            self.is_loaded = True
    
    def convert_voice(self, input_audio, pitch_shift: float = 0, 
                     conversion_strength: float = 0.8, 
                     formant_shift: float = 0.0) -> bytes:
        """
        Mock voice conversion process
        
        Args:
            input_audio: Input audio file object
            pitch_shift: Pitch adjustment in semitones
            conversion_strength: How much to apply the conversion (0-1)
            formant_shift: Formant adjustment
            
        Returns:
            bytes: Mock converted audio data
        """
        self.load_model()
        
        # Simulate processing time based on file size
        if hasattr(input_audio, 'size'):
            processing_time = min(max(input_audio.size / (1024 * 1024) * 0.5, 1.0), 10.0)
        else:
            processing_time = 2.0
        
        time.sleep(processing_time)
        
        # Generate mock audio data (silent WAV file with proper header)
        sample_rate = 44100
        duration = 5.0  # 5 seconds of mock audio
        samples = int(sample_rate * duration)
        
        # Create mock audio data with some variation based on parameters
        audio_data = np.zeros(samples, dtype=np.int16)
        
        # Add some mock processing effects based on parameters
        if pitch_shift != 0:
            # Simulate pitch shift by adding a slight frequency modulation
            t = np.linspace(0, duration, samples)
            freq = 440 + (pitch_shift * 10)  # Base frequency + pitch adjustment
            audio_data = (np.sin(2 * np.pi * freq * t) * 1000 * conversion_strength).astype(np.int16)
        
        # Create WAV file in memory
        wav_data = self._create_wav_bytes(audio_data, sample_rate)
        return wav_data
    
    def _create_wav_bytes(self, audio_data: np.ndarray, sample_rate: int) -> bytes:
        """Create WAV file bytes from audio data"""
        # WAV file header
        wav_header = bytearray()
        
        # RIFF chunk
        wav_header.extend(b'RIFF')
        wav_header.extend((36 + len(audio_data) * 2).to_bytes(4, 'little'))
        wav_header.extend(b'WAVE')
        
        # Format chunk
        wav_header.extend(b'fmt ')
        wav_header.extend((16).to_bytes(4, 'little'))  # Chunk size
        wav_header.extend((1).to_bytes(2, 'little'))   # Audio format (PCM)
        wav_header.extend((1).to_bytes(2, 'little'))   # Channels
        wav_header.extend(sample_rate.to_bytes(4, 'little'))
        wav_header.extend((sample_rate * 2).to_bytes(4, 'little'))  # Byte rate
        wav_header.extend((2).to_bytes(2, 'little'))   # Block align
        wav_header.extend((16).to_bytes(2, 'little'))  # Bits per sample
        
        # Data chunk
        wav_header.extend(b'data')
        wav_header.extend((len(audio_data) * 2).to_bytes(4, 'little'))
        
        # Combine header with audio data
        return bytes(wav_header) + audio_data.tobytes()

class MockVocalSeparator:
    """Mock vocal separation processor using UVR5 models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_loaded = False
    
    def load_model(self):
        """Simulate UVR5 model loading"""
        if not self.is_loaded:
            time.sleep(1.5)  # UVR5 models take a bit longer to load
            self.is_loaded = True
    
    def separate_vocals(self, input_audio) -> Tuple[bytes, bytes]:
        """
        Mock vocal separation process
        
        Args:
            input_audio: Input audio file object
            
        Returns:
            Tuple[bytes, bytes]: Mock separated vocals and instrumental audio data
        """
        self.load_model()
        
        # Simulate processing time based on file size
        if hasattr(input_audio, 'size'):
            processing_time = min(max(input_audio.size / (1024 * 1024) * 1.0, 2.0), 15.0)
        else:
            processing_time = 3.0
        
        time.sleep(processing_time)
        
        # Generate mock separated audio
        sample_rate = 44100
        duration = 5.0
        samples = int(sample_rate * duration)
        
        # Mock vocals (higher frequency content)
        t = np.linspace(0, duration, samples)
        vocals_data = (np.sin(2 * np.pi * 800 * t) * 0.7 * 1000).astype(np.int16)
        
        # Mock instrumental (lower frequency content)
        instrumental_data = (np.sin(2 * np.pi * 200 * t) * 0.5 * 1000).astype(np.int16)
        
        # Create WAV files
        vocals_wav = self._create_wav_bytes(vocals_data, sample_rate)
        instrumental_wav = self._create_wav_bytes(instrumental_data, sample_rate)
        
        return vocals_wav, instrumental_wav
    
    def _create_wav_bytes(self, audio_data: np.ndarray, sample_rate: int) -> bytes:
        """Create WAV file bytes from audio data"""
        # WAV file header (same as in MockVoiceConverter)
        wav_header = bytearray()
        
        # RIFF chunk
        wav_header.extend(b'RIFF')
        wav_header.extend((36 + len(audio_data) * 2).to_bytes(4, 'little'))
        wav_header.extend(b'WAVE')
        
        # Format chunk
        wav_header.extend(b'fmt ')
        wav_header.extend((16).to_bytes(4, 'little'))
        wav_header.extend((1).to_bytes(2, 'little'))
        wav_header.extend((1).to_bytes(2, 'little'))
        wav_header.extend(sample_rate.to_bytes(4, 'little'))
        wav_header.extend((sample_rate * 2).to_bytes(4, 'little'))
        wav_header.extend((2).to_bytes(2, 'little'))
        wav_header.extend((16).to_bytes(2, 'little'))
        
        # Data chunk
        wav_header.extend(b'data')
        wav_header.extend((len(audio_data) * 2).to_bytes(4, 'little'))
        
        return bytes(wav_header) + audio_data.tobytes()

class MockModelTrainer:
    """Mock model trainer for RVC models"""
    
    def __init__(self):
        self.training_status = "ready"
        self.current_epoch = 0
        self.total_epochs = 0
        
    def prepare_dataset(self, audio_files):
        """Mock dataset preparation"""
        time.sleep(1)
        return True
    
    def start_training(self, model_name: str, speaker_name: str, 
                      epochs: int = 500, batch_size: int = 16,
                      learning_rate: float = 0.001):
        """
        Mock training process
        
        Args:
            model_name: Name for the new model
            speaker_name: Speaker identifier
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
            
        Returns:
            dict: Training configuration
        """
        self.training_status = "training"
        self.current_epoch = 0
        self.total_epochs = epochs
        
        return {
            'model_name': model_name,
            'speaker_name': speaker_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'status': 'started'
        }
    
    def get_training_progress(self):
        """Get current training progress"""
        if self.training_status == "training":
            progress = min(self.current_epoch / self.total_epochs * 100, 100)
            return {
                'progress': progress,
                'current_epoch': self.current_epoch,
                'total_epochs': self.total_epochs,
                'status': self.training_status
            }
        return {'progress': 0, 'status': self.training_status}
    
    def update_progress(self):
        """Mock progress update"""
        if self.training_status == "training" and self.current_epoch < self.total_epochs:
            self.current_epoch += 1
            if self.current_epoch >= self.total_epochs:
                self.training_status = "completed"
    
    def export_model(self, model_name: str) -> bytes:
        """Mock model export"""
        # Create mock model file
        model_data = f"Mock RVC Model: {model_name}\nTrained epochs: {self.total_epochs}\nTimestamp: {time.time()}"
        return model_data.encode('utf-8')

class MockModelManager:
    """Mock model management for RVC models"""
    
    def __init__(self):
        self.available_models = [
            "Default RVC Model",
            "High Quality Singer", 
            "Speech Enhancement",
            "Anime Character Voice",
            "Professional Narrator"
        ]
        self.loaded_models = {}
    
    def upload_model(self, model_file, model_name: str) -> bool:
        """
        Mock model upload process
        
        Args:
            model_file: Uploaded model file
            model_name: Name for the model
            
        Returns:
            bool: Success status
        """
        # Simulate upload processing
        time.sleep(2)
        
        if model_file.name.endswith('.zip'):
            self.available_models.append(model_name)
            return True
        return False
    
    def get_model_info(self, model_name: str) -> dict:
        """Get information about a model"""
        return {
            'name': model_name,
            'type': 'RVC Model',
            'status': 'Available',
            'size': '50MB',
            'quality': 'High',
            'language': 'Multi-language'
        }
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a custom model"""
        if model_name in self.available_models and model_name not in [
            "Default RVC Model", "High Quality Singer", "Speech Enhancement",
            "Anime Character Voice", "Professional Narrator"
        ]:
            self.available_models.remove(model_name)
            return True
        return False
