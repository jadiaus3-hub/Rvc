# RVC-GUI Voice Conversion Interface

## Overview

This is a Streamlit-based web application that clones the RVC (Retrieval-based Voice Conversion) GUI interface. The application provides a user-friendly interface for voice conversion and vocal separation tasks, featuring model management, audio file processing, and real-time parameter adjustment. The current implementation uses mock AI processing to simulate voice conversion and vocal separation functionality, making it ideal for prototyping and UI/UX development.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid prototyping and deployment
- **UI Components**: Uses Streamlit's built-in widgets (file uploaders, sliders, selectboxes, progress bars)
- **Layout**: Wide layout with expandable sidebar for model management and main content area for audio processing
- **State Management**: Streamlit session state for persisting processed audio, conversion history, and separated audio tracks

### Backend Architecture
- **Processing Layer**: Mock processing classes (`MockVoiceConverter`, `MockVocalSeparator`) that simulate AI model behavior
- **Audio Handling**: Utility functions for audio file validation, format checking, and metadata extraction
- **File Management**: Temporary file handling for audio uploads and processing with size limitations (100MB)
- **Model System**: Support for multiple pre-trained models and custom model uploads via ZIP files

### Data Flow Design
- **Input Validation**: Multi-stage validation including file format checking and size limits
- **Processing Pipeline**: Simulated AI processing with realistic timing based on file size
- **Output Generation**: Mock audio generation with proper WAV headers and metadata

### Session Management
- **Persistent State**: Audio processing results and conversion history maintained across user interactions
- **Memory Management**: Efficient handling of audio data in session state to prevent memory leaks

## External Dependencies

### Core Dependencies
- **Streamlit**: Web application framework for the user interface
- **NumPy**: Numerical computing for audio data manipulation and mock signal generation
- **Python Standard Library**: `tempfile`, `os`, `time`, `io` for file operations and system utilities

### Audio Processing (Planned)
- The architecture is designed to easily integrate real audio processing libraries such as:
  - **librosa**: For audio analysis and feature extraction
  - **pydub**: For audio format conversion and basic manipulation
  - **torch/tensorflow**: For actual AI model inference when replacing mock implementations

### File Format Support
- Supports multiple audio formats: WAV, MP3, FLAC, M4A, OGG, AAC
- ZIP file support for custom model uploads
- Built-in file size validation and format checking

### Deployment Considerations
- Designed for easy deployment on Streamlit Cloud, Heroku, or similar platforms
- No external database requirements - uses in-memory session state
- Minimal external dependencies for rapid prototyping and testing