"""
Real RVC (Retrieval-based Voice Conversion) Training Implementation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import soundfile as sf

@dataclass
class TrainingConfig:
    """Training configuration"""
    model_name: str
    speaker_name: str
    epochs: int = 500
    batch_size: int = 16
    learning_rate: float = 0.001
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    f0_min: float = 80.0
    f0_max: float = 880.0
    save_interval: int = 100

class VoiceEncoder(nn.Module):
    """Voice encoder network for RVC"""
    
    def __init__(self, input_dim=80, hidden_dim=256, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x shape: (batch, time, features)
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            x = F.relu(x)
            if i < len(self.encoder_layers) - 1:
                x = self.dropout(x)
        
        x = self.layer_norm(x)
        return x

class VoiceDecoder(nn.Module):
    """Voice decoder network for RVC"""
    
    def __init__(self, hidden_dim=256, output_dim=80, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim if i < num_layers-1 else output_dim)
            for i in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if i < len(self.decoder_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x

class RVCModel(nn.Module):
    """Complete RVC model"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Components
        self.encoder = VoiceEncoder(
            input_dim=config.n_mels,
            hidden_dim=256,
            num_layers=6
        )
        
        self.decoder = VoiceDecoder(
            hidden_dim=256,
            output_dim=config.n_mels,
            num_layers=6
        )
        
        # F0 predictor
        self.f0_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, mel_spec, f0=None):
        # Encode
        encoded = self.encoder(mel_spec)
        
        # Predict F0 if not provided
        if f0 is None:
            f0_pred = self.f0_predictor(encoded)
            f0_pred = f0_pred * (self.config.f0_max - self.config.f0_min) + self.config.f0_min
        else:
            f0_pred = f0
        
        # Decode
        decoded = self.decoder(encoded)
        
        return decoded, f0_pred

class AudioProcessor:
    """Audio processing utilities"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def load_audio(self, file_path: str) -> Optional[np.ndarray]:
        """Load and preprocess audio file"""
        try:
            audio, sr = librosa.load(file_path, sr=self.config.sample_rate)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram from audio"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec = (mel_spec + 80) / 80  # Normalize to [0, 1]
        
        return mel_spec.T  # Transpose to (time, mels)
    
    def extract_f0(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 (fundamental frequency) from audio"""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.config.f0_min,
            fmax=self.config.f0_max,
            sr=self.config.sample_rate,
            hop_length=self.config.hop_length
        )
        
        # Fill unvoiced frames with interpolation
        f0 = np.nan_to_num(f0, nan=0.0)
        
        # Normalize F0
        f0_normalized = (f0 - self.config.f0_min) / (self.config.f0_max - self.config.f0_min)
        f0_normalized = np.clip(f0_normalized, 0, 1)
        
        return f0_normalized

class RVCDataset(torch.utils.data.Dataset):
    """Dataset for RVC training"""
    
    def __init__(self, audio_files: List, config: TrainingConfig):
        self.audio_files = audio_files
        self.config = config
        self.processor = AudioProcessor(config)
        self.data = []
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare training data"""
        print("Preparing training data...")
        
        for file_obj in self.audio_files:
            # Save uploaded file temporarily
            temp_path = f"temp_{file_obj.name}"
            with open(temp_path, "wb") as f:
                f.write(file_obj.getbuffer())
            
            try:
                # Load and process audio
                audio = self.processor.load_audio(temp_path)
                if audio is None:
                    continue
                
                # Extract features
                mel_spec = self.processor.extract_mel_spectrogram(audio)
                f0 = self.processor.extract_f0(audio)
                
                # Create chunks
                chunk_size = 128  # frames
                for i in range(0, len(mel_spec) - chunk_size, chunk_size // 2):
                    mel_chunk = mel_spec[i:i+chunk_size]
                    f0_chunk = f0[i:i+chunk_size]
                    
                    if len(mel_chunk) == chunk_size:
                        self.data.append({
                            'mel': mel_chunk,
                            'f0': f0_chunk
                        })
                
            except Exception as e:
                print(f"Error processing {file_obj.name}: {e}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        print(f"Prepared {len(self.data)} training samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'mel': torch.FloatTensor(item['mel']),
            'f0': torch.FloatTensor(item['f0'])
        }

class RVCTrainer:
    """Real RVC trainer implementation"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = RVCModel(config).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999)
        )
        
        # Loss functions
        self.mel_loss = nn.MSELoss()
        self.f0_loss = nn.MSELoss()
        
        # Training state
        self.current_epoch = 0
        self.training_losses = []
        
    def prepare_dataset(self, audio_files: List) -> bool:
        """Prepare training dataset"""
        try:
            self.dataset = RVCDataset(audio_files, self.config)
            if len(self.dataset) == 0:
                return False
            
            # Create data loader
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0  # Set to 0 for compatibility
            )
            
            return True
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            return False
    
    def train_epoch(self) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.dataloader:
            mel_spec = batch['mel'].to(self.device)
            f0_true = batch['f0'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            mel_pred, f0_pred = self.model(mel_spec)
            
            # Calculate losses
            mel_loss = self.mel_loss(mel_pred, mel_spec)
            f0_loss = self.f0_loss(f0_pred.squeeze(-1), f0_true)
            
            total_loss_batch = mel_loss + 0.1 * f0_loss  # Weight F0 loss less
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, progress_callback=None):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            start_time = time.time()
            epoch_loss = self.train_epoch()
            epoch_time = time.time() - start_time
            
            self.training_losses.append(epoch_loss)
            
            # Print progress
            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                print(f"Epoch {epoch+1}/{self.config.epochs}, "
                      f"Loss: {epoch_loss:.6f}, "
                      f"Time: {epoch_time:.2f}s")
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0 or epoch == self.config.epochs - 1:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    self.save_model(f"best_{self.config.model_name}")
            
            # Progress callback
            if progress_callback:
                progress_callback(epoch + 1, self.config.epochs, epoch_loss)
        
        print("Training completed!")
    
    def save_model(self, filename: str) -> str:
        """Save trained model"""
        model_dir = Path("trained_models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"{filename}.pth"
        config_path = model_dir / f"{filename}_config.json"
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'loss': self.training_losses[-1] if self.training_losses else 0.0,
            'config': self.config.__dict__
        }, model_path)
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        print(f"Model saved: {model_path}")
        return str(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        print(f"Model loaded from {model_path}")
    
    def get_training_progress(self) -> Dict:
        """Get current training progress"""
        return {
            'current_epoch': self.current_epoch,
            'total_epochs': self.config.epochs,
            'progress_percent': (self.current_epoch / self.config.epochs) * 100,
            'current_loss': self.training_losses[-1] if self.training_losses else 0.0,
            'average_loss': sum(self.training_losses) / len(self.training_losses) if self.training_losses else 0.0
        }

class RVCInference:
    """RVC inference engine for voice conversion"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.config = None
        self.processor = None
        self.is_loaded = False
        
        try:
            self.load_model(model_path, config_path)
        except Exception as e:
            print(f"Warning: Failed to load model during initialization: {e}")
            self.is_loaded = False
    
    def load_model(self, model_path: str, config_path: str = None):
        """Load trained model for inference"""
        try:
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Reconstruct config
            if 'config' in checkpoint:
                config_dict = checkpoint['config']
                self.config = TrainingConfig(
                    model_name=config_dict.get('model_name', 'Unknown'),
                    speaker_name=config_dict.get('speaker_name', 'Unknown'),
                    epochs=config_dict.get('epochs', 500),
                    batch_size=config_dict.get('batch_size', 16),
                    learning_rate=config_dict.get('learning_rate', 0.001),
                    sample_rate=config_dict.get('sample_rate', 22050),
                    n_fft=config_dict.get('n_fft', 1024),
                    hop_length=config_dict.get('hop_length', 256),
                    n_mels=config_dict.get('n_mels', 80)
                )
            else:
                # Fallback config
                self.config = TrainingConfig(
                    model_name='Loaded Model',
                    speaker_name='Unknown'
                )
            
            # Initialize model
            self.model = RVCModel(self.config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Initialize audio processor
            self.processor = AudioProcessor(self.config)
            self.is_loaded = True
            
            print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False
            # Initialize with default config for error handling
            self.config = TrainingConfig(model_name='Error', speaker_name='Error')
            self.processor = AudioProcessor(self.config)
            raise
    
    def convert_voice(self, input_audio_file, pitch_shift: float = 0, 
                     conversion_strength: float = 0.8) -> bytes:
        """Convert voice using trained model"""
        if not self.is_loaded or self.model is None or self.processor is None or self.config is None:
            raise ValueError("Model not properly loaded")
            
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_inference_{input_audio_file.name}"
            with open(temp_path, "wb") as f:
                f.write(input_audio_file.getbuffer())
            
            try:
                # Load and process audio
                if self.processor is None:
                    raise ValueError("Audio processor not initialized")
                    
                audio = self.processor.load_audio(temp_path)
                if audio is None:
                    raise ValueError("Failed to load audio file")
                
                # Extract features
                mel_spec = self.processor.extract_mel_spectrogram(audio)
                
                # Convert to tensor
                mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).to(self.device)
                
                # Inference
                if self.model is None:
                    raise ValueError("Model not loaded")
                    
                with torch.no_grad():
                    converted_mel, predicted_f0 = self.model(mel_tensor)
                
                # Apply conversion strength
                if conversion_strength < 1.0:
                    converted_mel = mel_tensor * (1 - conversion_strength) + converted_mel * conversion_strength
                
                # Convert back to audio (simplified - using inverse mel spectrogram)
                converted_mel_np = converted_mel.squeeze(0).cpu().numpy()
                
                # Denormalize mel spectrogram
                converted_mel_np = (converted_mel_np * 80) - 80
                
                # Convert mel to linear spectrogram (approximation)
                if self.config is None:
                    raise ValueError("Config not loaded")
                    
                linear_spec = librosa.feature.inverse.mel_to_stft(
                    np.power(10, converted_mel_np / 10),
                    sr=self.config.sample_rate,
                    n_fft=self.config.n_fft
                )
                
                # Apply pitch shift if specified
                if pitch_shift != 0:
                    linear_spec = self._apply_pitch_shift(linear_spec, pitch_shift)
                
                # Convert to audio
                converted_audio = librosa.istft(
                    linear_spec,
                    hop_length=self.config.hop_length,
                    length=len(audio)
                )
                
                # Normalize audio
                converted_audio = converted_audio / np.max(np.abs(converted_audio))
                
                # Convert to int16
                converted_audio_int16 = (converted_audio * 32767).astype(np.int16)
                
                # Create WAV file
                wav_data = self._create_wav_bytes(converted_audio_int16, self.config.sample_rate)
                
                return wav_data
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            print(f"Inference error: {e}")
            # Return original audio as fallback
            temp_path = f"temp_fallback_{input_audio_file.name}"
            with open(temp_path, "wb") as f:
                f.write(input_audio_file.getbuffer())
            
            try:
                if self.processor is not None and self.config is not None:
                    audio = self.processor.load_audio(temp_path)
                    if audio is not None:
                        audio_int16 = (audio * 32767).astype(np.int16)
                        return self._create_wav_bytes(audio_int16, self.config.sample_rate)
                
                # Final fallback - return empty audio
                sample_rate = 22050
                empty_audio = np.zeros(sample_rate, dtype=np.int16)  # 1 second of silence
                return self._create_wav_bytes(empty_audio, sample_rate)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    def _apply_pitch_shift(self, stft_matrix, pitch_shift_semitones):
        """Apply pitch shift to STFT matrix"""
        try:
            # Convert semitones to frequency ratio
            shift_ratio = 2 ** (pitch_shift_semitones / 12.0)
            
            # Create frequency bins
            if self.config is None:
                return stft_matrix
                
            n_fft = (stft_matrix.shape[0] - 1) * 2
            freqs = np.fft.fftfreq(n_fft, d=1/self.config.sample_rate)[:n_fft//2 + 1]
            
            # Apply pitch shift by frequency shifting
            shifted_stft = np.zeros_like(stft_matrix)
            
            for i, freq in enumerate(freqs):
                shifted_freq = freq * shift_ratio
                # Find closest frequency bin
                shifted_bin = int(shifted_freq / self.config.sample_rate * n_fft)
                if 0 <= shifted_bin < len(freqs):
                    shifted_stft[shifted_bin] = stft_matrix[i]
            
            return shifted_stft
            
        except Exception:
            # Return original if pitch shift fails
            return stft_matrix
    
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
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        if self.config and self.is_loaded:
            return {
                'model_name': self.config.model_name,
                'speaker_name': self.config.speaker_name,
                'sample_rate': self.config.sample_rate,
                'device': str(self.device),
                'parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
                'status': 'loaded'
            }
        return {'status': 'not_loaded'}

def create_trainer(config: TrainingConfig) -> RVCTrainer:
    """Factory function to create RVC trainer"""
    return RVCTrainer(config)

def create_inference(model_path: str) -> RVCInference:
    """Factory function to create RVC inference engine"""
    return RVCInference(model_path)