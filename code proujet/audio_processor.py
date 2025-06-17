import whisper
import torch
from pydub import AudioSegment
import os
import logging
import tempfile
from typing import Optional, Dict, List, Union

class AudioProcessor:
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize Whisper model for speech-to-text
        
        Args:
            model_size: Size of the Whisper model ("tiny", "base", "small", "medium", "large")
            device: Device to run the model on ("cuda", "cpu"). If None, automatically selects.
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('AudioProcessor')
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        try:
            self.model_size = model_size
            self.logger.info(f"Loading Whisper {model_size} model...")
            self.model = whisper.load_model(model_size, device=self.device)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {str(e)}")
            raise
    
    def transcribe_audio(self, 
                         audio_path: str, 
                         language: Optional[str] = None,
                         task: str = "transcribe",
                         verbose: bool = False,
                         **kwargs) -> Dict:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file (.mp3, .wav, etc.)
            language: Language code for transcription (e.g., "en", "fr"). None for auto-detection.
            task: Either "transcribe" or "translate" (to English)
            verbose: Whether to print progress information
            **kwargs: Additional arguments to pass to whisper's transcribe function
        
        Returns:
            Dictionary containing transcription results
        """
        try:
            # Validate input file
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Convert to compatible format if needed
            temp_file = None
            processed_path = audio_path
            
            if not audio_path.lower().endswith('.wav'):
                self.logger.info(f"Converting {audio_path} to WAV format")
                try:
                    audio = AudioSegment.from_file(audio_path)
                    # Use a temporary file to avoid modifying the original directory
                    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    temp_path = temp_file.name
                    temp_file.close()  # Close but don't delete
                    
                    audio.export(temp_path, format="wav")
                    processed_path = temp_path
                    self.logger.info(f"Converted to WAV: {processed_path}")
                except Exception as e:
                    self.logger.error(f"Failed to convert audio format: {str(e)}")
                    if temp_file and os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                    raise
            
            # Prepare transcription options
            options = {
                "task": task,
                "verbose": verbose
            }
            
            if language:
                options["language"] = language
                
            # Add any additional kwargs
            options.update(kwargs)
                
            # Transcribe
            self.logger.info(f"Transcribing audio with options: {options}")
            result = self.model.transcribe(processed_path, **options)
            
            # Clean up temporary file if created
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
                
            return result
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            # Clean up temporary file in case of exception
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise
    
    def transcribe_to_text(self, 
                          audio_path: str, 
                          language: Optional[str] = None) -> str:
        """
        Simple wrapper to get just the text from transcription
        
        Args:
            audio_path: Path to audio file
            language: Language code for transcription
            
        Returns:
            Transcribed text string
        """
        result = self.transcribe_audio(audio_path, language=language)
        return result["text"]
    
    def batch_transcribe(self, 
                        file_list: List[str], 
                        language: Optional[str] = None) -> Dict[str, str]:
        """
        Transcribe multiple audio files
        
        Args:
            file_list: List of paths to audio files
            language: Language code for transcription
            
        Returns:
            Dictionary mapping file paths to their transcriptions
        """
        results = {}
        for file_path in file_list:
            try:
                self.logger.info(f"Processing file: {file_path}")
                results[file_path] = self.transcribe_to_text(file_path, language)
            except Exception as e:
                self.logger.error(f"Failed to transcribe {file_path}: {str(e)}")
                results[file_path] = f"ERROR: {str(e)}"
        
        return results
    
    def transcribe_with_timestamps(self, 
                                 audio_path: str, 
                                 language: Optional[str] = None) -> Dict:
        """
        Transcribe audio and return segments with timestamps
        
        Args:
            audio_path: Path to audio file
            language: Language code for transcription
            
        Returns:
            Dictionary with segments containing text and timestamps
        """
        result = self.transcribe_audio(audio_path, language=language)
        return {
            "text": result["text"],
            "segments": result["segments"]
        }
    
    def get_available_languages(self) -> List[str]:
        """
        Get list of available languages in Whisper
        
        Returns:
            List of language codes supported by the model
        """
        return list(whisper.tokenizer.LANGUAGES.keys())
        
    @staticmethod
    def supported_formats() -> List[str]:
        """
        List supported audio formats
        
        Returns:
            List of supported audio file extensions
        """
        return ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma']
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model details
        """
        return {
            "model_size": self.model_size,
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "multilingual": self.model.is_multilingual
        }
