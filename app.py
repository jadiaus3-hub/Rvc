import streamlit as st
import time
import os
import numpy as np
from audio_utils import validate_audio_file, get_audio_info
from mock_processing import MockVoiceConverter, MockVocalSeparator
from rvc_trainer import RVCTrainer, TrainingConfig, create_trainer
import tempfile

# Initialize session state
if 'processed_audio' not in st.session_state:
    st.session_state.processed_audio = None
if 'separated_vocals' not in st.session_state:
    st.session_state.separated_vocals = None
if 'separated_instrumental' not in st.session_state:
    st.session_state.separated_instrumental = None
if 'conversion_history' not in st.session_state:
    st.session_state.conversion_history = []
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = []
if 'rvc_trainer' not in st.session_state:
    st.session_state.rvc_trainer = None
if 'training_logs' not in st.session_state:
    st.session_state.training_logs = []

# Page configuration
st.set_page_config(
    page_title="RVC-GUI Clone",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🎤 RVC-GUI Voice Conversion Interface")
st.markdown("*Streamlit clone with mocked AI processing functionality*")

# Sidebar for model management
with st.sidebar:
    st.header("🔧 Model Management")
    
    # Model selection (include trained models)
    available_models = [
        "Default RVC Model",
        "High Quality Singer",
        "Speech Enhancement",
        "Anime Character Voice",
        "Professional Narrator"
    ]
    
    # Add trained models to the list
    for trained_model in st.session_state.trained_models:
        if trained_model['name'] not in available_models:
            available_models.append(f"{trained_model['name']} (Custom)")
    selected_model = st.selectbox("Select Voice Model", available_models)
    
    # Model upload
    st.subheader("Upload Custom Model")
    uploaded_model = st.file_uploader(
        "Upload RVC Model (.zip)", 
        type=['zip'],
        help="Upload a pre-trained RVC model in .zip format"
    )
    
    if uploaded_model:
        st.success(f"Model uploaded: {uploaded_model.name}")
        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                time.sleep(2)  # Mock loading time
            st.success("Model loaded successfully!")
    
    # Show trained models section
    if st.session_state.trained_models:
        st.subheader("🏆 Your Trained Models")
        for i, model in enumerate(st.session_state.trained_models):
            with st.expander(f"🎤 {model['name']}"):
                st.write(f"**Speaker:** {model['speaker']}")
                st.write(f"**Epochs:** {model['epochs']}")
                st.write(f"**Created:** {model['created']}")
                
                # Download real trained model
                if 'model_path' in model and os.path.exists(model['model_path']):
                    with open(model['model_path'], 'rb') as f:
                        model_file_data = f.read()
                    
                    st.download_button(
                        f"📥 Download {model['name']}",
                        data=model_file_data,
                        file_name=f"{model['name'].replace(' ', '_')}_trained.pth",
                        mime="application/octet-stream",
                        key=f"download_{i}",
                        help=f"โมเดลที่เทรนจริง - {len(model_file_data) / 1024 / 1024:.1f} MB"
                    )
                else:
                    st.warning("⚠️ Model file not found")
                    st.info("โมเดลที่เทรนเสร็จแล้ว กรุณาเทรนใหม่อีกครั้ง")
    
    # Processing settings
    st.subheader("⚙️ Processing Settings")
    
    # Hardware selection
    hardware_option = st.selectbox(
        "Hardware",
        ["CPU (Free)", "GPU (Faster)", "Cloud Processing"]
    )
    
    # Quality settings
    quality = st.select_slider(
        "Processing Quality",
        options=["Fast", "Balanced", "High Quality"],
        value="Balanced"
    )

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎵 Voice Conversion", 
    "🎼 Vocal Separation", 
    "🏋️ Model Training",
    "📊 Processing History",
    "ℹ️ About"
])

with tab1:
    st.header("Voice Conversion")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Input Audio")
        
        # Audio input methods
        input_method = st.radio(
            "Choose input method:",
            ["Upload File", "Record Audio"]
        )
        
        input_audio = None
        
        if input_method == "Upload File":
            input_audio = st.file_uploader(
                "Upload audio file",
                type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
                help="Supported formats: WAV, MP3, FLAC, M4A, OGG"
            )
            
            if input_audio:
                if validate_audio_file(input_audio):
                    st.success("✅ Valid audio file uploaded")
                    audio_info = get_audio_info(input_audio)
                    st.info(f"📋 Duration: {audio_info['duration']:.1f}s | Format: {audio_info['format']}")
                    st.audio(input_audio)
                else:
                    st.error("❌ Invalid audio file format")
        
        else:  # Record Audio
            st.markdown("🎙️ **Voice Recording**")
            
            # Simple recording interface (mocked)
            if st.button("🔴 Start Recording"):
                st.session_state.recording = True
                st.rerun()
            
            if st.session_state.get('recording', False):
                recording_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                for i in range(101):
                    progress_bar.progress(i)
                    recording_placeholder.text(f"Recording... {i/10:.1f}s")
                    time.sleep(0.1)
                
                st.session_state.recording = False
                st.success("✅ Recording completed!")
                
                # Mock recorded audio
                st.audio("data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+Dt")
        
        # Conversion parameters
        st.subheader("🎛️ Conversion Parameters")
        
        pitch_shift = st.slider(
            "Pitch Shift",
            min_value=-12,
            max_value=12,
            value=0,
            help="Adjust pitch in semitones"
        )
        
        conversion_strength = st.slider(
            "Conversion Strength",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            help="How much to apply the voice conversion"
        )
        
        formant_shift = st.slider(
            "Formant Shift",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            help="Adjust vocal tract characteristics"
        )
        
        # Advanced settings expander
        with st.expander("🔧 Advanced Settings"):
            harvest_median_filter = st.slider("Harvest Median Filter", 0, 7, 3)
            resample_sr = st.selectbox("Resample Rate", [0, 16000, 22050, 44100, 48000])
            envelope_mix = st.slider("Envelope Mix", 0.0, 1.0, 1.0)
    
    with col2:
        st.subheader("📥 Output Audio")
        
        # Convert button
        if st.button("🚀 Convert Voice", type="primary", disabled=input_audio is None):
            if input_audio:
                # Processing simulation
                progress_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                converter = MockVoiceConverter(selected_model)
                
                stages = [
                    ("Preprocessing audio...", 20),
                    ("Loading voice model...", 40),
                    ("Extracting features...", 60),
                    ("Converting voice...", 80),
                    ("Post-processing...", 100)
                ]
                
                for stage_text, progress in stages:
                    status_text.text(stage_text)
                    progress_bar.progress(progress)
                    time.sleep(1.5)  # Realistic processing delay
                
                # Mock conversion result
                st.session_state.processed_audio = converter.convert_voice(
                    input_audio,
                    pitch_shift,
                    conversion_strength,
                    formant_shift
                )
                
                # Add to history
                st.session_state.conversion_history.append({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'model': selected_model,
                    'parameters': {
                        'pitch_shift': pitch_shift,
                        'conversion_strength': conversion_strength,
                        'formant_shift': formant_shift
                    }
                })
                
                status_text.text("✅ Conversion completed!")
                progress_bar.progress(100)
                st.success("Voice conversion completed successfully!")
                st.rerun()
        
        # Display processed audio
        if st.session_state.processed_audio:
            st.success("🎉 Conversion Complete!")
            st.audio(st.session_state.processed_audio)
            
            # Download button
            st.download_button(
                label="📥 Download Converted Audio",
                data=st.session_state.processed_audio,
                file_name=f"converted_voice_{int(time.time())}.wav",
                mime="audio/wav"
            )
            
            # Comparison player
            if input_audio:
                st.subheader("🔄 Before/After Comparison")
                comp_col1, comp_col2 = st.columns(2)
                with comp_col1:
                    st.write("**Original**")
                    st.audio(input_audio)
                with comp_col2:
                    st.write("**Converted**")
                    st.audio(st.session_state.processed_audio)

with tab2:
    st.header("Vocal Separation")
    st.markdown("Separate vocals from instrumental using UVR5 models")
    
    sep_col1, sep_col2 = st.columns([1, 1])
    
    with sep_col1:
        st.subheader("📤 Input Audio")
        
        separation_audio = st.file_uploader(
            "Upload audio for separation",
            type=['wav', 'mp3', 'flac', 'm4a'],
            key="separation_upload"
        )
        
        # UVR5 model selection
        uvr_models = [
            "UVR-DeEcho-DeReverb",
            "UVR5_Vocals_Model",
            "Kim_Vocal_2",
            "UVR-BVE-4B_SN-44100-1"
        ]
        selected_uvr_model = st.selectbox("Select UVR5 Model", uvr_models)
        
        if separation_audio:
            st.audio(separation_audio)
            
            # Separation settings
            st.subheader("🎛️ Separation Settings")
            
            aggressive_separation = st.checkbox("Aggressive Separation", value=False)
            output_format = st.selectbox("Output Format", ["WAV", "FLAC", "MP3"])
            
    with sep_col2:
        st.subheader("📥 Separated Audio")
        
        if st.button("🎵 Separate Vocals", type="primary", disabled=separation_audio is None):
            if separation_audio:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                separator = MockVocalSeparator(selected_uvr_model)
                
                separation_stages = [
                    ("Analyzing audio structure...", 25),
                    ("Loading UVR5 model...", 50),
                    ("Separating vocals...", 75),
                    ("Finalizing outputs...", 100)
                ]
                
                for stage_text, progress in separation_stages:
                    status_text.text(stage_text)
                    progress_bar.progress(progress)
                    time.sleep(2)
                
                vocals, instrumental = separator.separate_vocals(separation_audio)
                st.session_state.separated_vocals = vocals
                st.session_state.separated_instrumental = instrumental
                
                status_text.text("✅ Separation completed!")
                st.success("Vocal separation completed successfully!")
                st.rerun()
        
        # Display separated audio
        if st.session_state.separated_vocals and st.session_state.separated_instrumental:
            st.success("🎉 Separation Complete!")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.write("**🎤 Vocals Only**")
                st.audio(st.session_state.separated_vocals)
                st.download_button(
                    "📥 Download Vocals",
                    data=st.session_state.separated_vocals,
                    file_name=f"vocals_{int(time.time())}.wav",
                    mime="audio/wav"
                )
            
            with result_col2:
                st.write("**🎼 Instrumental Only**")
                st.audio(st.session_state.separated_instrumental)
                st.download_button(
                    "📥 Download Instrumental",
                    data=st.session_state.separated_instrumental,
                    file_name=f"instrumental_{int(time.time())}.wav",
                    mime="audio/wav"
                )

with tab3:
    st.header("Model Training")
    st.markdown("Train your own RVC voice models with custom datasets")
    
    training_col1, training_col2 = st.columns([1, 1])
    
    with training_col1:
        st.subheader("📁 Dataset Preparation")
        
        # Dataset upload
        dataset_files = st.file_uploader(
            "Upload Training Audio Files",
            type=['wav', 'mp3', 'flac'],
            accept_multiple_files=True,
            help="Upload multiple audio files (at least 10 minutes total recommended)"
        )
        
        if dataset_files:
            st.success(f"✅ {len(dataset_files)} files uploaded")
            
            total_duration = 0
            st.write("**Uploaded Files:**")
            for i, file in enumerate(dataset_files[:5]):  # Show first 5 files
                file_info = get_audio_info(file)
                total_duration += file_info['duration']
                st.write(f"• {file.name} - {file_info['duration']:.1f}s")
            
            if len(dataset_files) > 5:
                st.write(f"... และอีก {len(dataset_files) - 5} ไฟล์")
            
            st.info(f"📊 รวมความยาว: {total_duration/60:.1f} นาที")
            
            # Quality check
            if total_duration < 600:  # Less than 10 minutes
                st.warning("⚠️ แนะนำให้มีข้อมูลเสียงอย่างน้อย 10 นาที เพื่อผลลัพธ์ที่ดี")
            else:
                st.success("✅ ข้อมูลเสียงเพียงพอสำหรับการเทรน")
        
        # Model configuration
        st.subheader("⚙️ Model Configuration")
        
        model_name = st.text_input(
            "Model Name",
            placeholder="ตัวอย่าง: MyVoiceModel_v1",
            help="ชื่อโมเดลที่จะสร้าง"
        )
        
        speaker_name = st.text_input(
            "Speaker Name",
            placeholder="ตัวอย่าง: นาย ก",
            help="ชื่อผู้พูดสำหรับโมเดลนี้"
        )
        
        # Training parameters
        st.subheader("🎛️ Training Parameters")
        
        with st.expander("Advanced Training Settings"):
            epochs = st.slider("Training Epochs", 100, 1000, 500, help="จำนวนรอบการเทรน")
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)
            learning_rate = st.select_slider(
                "Learning Rate", 
                options=["0.0001", "0.0005", "0.001", "0.005"],
                value="0.001"
            )
            save_frequency = st.slider("Save Every N Epochs", 50, 200, 100)
    
    with training_col2:
        st.subheader("🚀 Training Process")
        
        # Initialize real training state
        if 'training_status' not in st.session_state:
            st.session_state.training_status = "ready"
        
        # Training button
        can_train = (dataset_files and len(dataset_files) > 0 and 
                    model_name.strip() != "" and speaker_name.strip() != "")
        
        if st.button("🏋️ Start Real Training", type="primary", disabled=not can_train):
            if can_train:
                try:
                    # Create training configuration
                    config = TrainingConfig(
                        model_name=model_name,
                        speaker_name=speaker_name,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=float(learning_rate)
                    )
                    
                    # Create trainer
                    trainer = create_trainer(config)
                    
                    # Prepare dataset
                    with st.spinner("📦 Preparing dataset..."):
                        if trainer.prepare_dataset(dataset_files):
                            st.session_state.rvc_trainer = trainer
                            st.session_state.training_status = "training"
                            st.session_state.training_logs = [f"Started training {model_name}"]
                            st.success("✅ Dataset prepared successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to prepare dataset. Please check your audio files.")
                
                except Exception as e:
                    st.error(f"❌ Training setup failed: {str(e)}")
                    st.session_state.training_status = "ready"
        
        # Real training status display
        if st.session_state.training_status == "training" and st.session_state.rvc_trainer:
            st.subheader("🔄 Real Training in Progress")
            
            trainer = st.session_state.rvc_trainer
            progress_info = trainer.get_training_progress()
            
            # Progress display
            progress_bar = st.progress(progress_info['progress_percent'] / 100)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Epoch", f"{progress_info['current_epoch']}/{progress_info['total_epochs']}")
            with col2:
                st.metric("Current Loss", f"{progress_info['current_loss']:.6f}")
            with col3:
                st.metric("Avg Loss", f"{progress_info['average_loss']:.6f}")
            
            # Training logs
            with st.expander("📝 Training Logs"):
                for log in st.session_state.training_logs[-10:]:  # Show last 10 logs
                    st.text(log)
            
            # Train one epoch at a time for UI responsiveness
            if progress_info['current_epoch'] < progress_info['total_epochs']:
                try:
                    with st.spinner(f"Training epoch {progress_info['current_epoch'] + 1}..."):
                        epoch_loss = trainer.train_epoch()
                        trainer.current_epoch += 1
                        
                        # Log progress
                        log_msg = f"Epoch {trainer.current_epoch}: Loss = {epoch_loss:.6f}"
                        st.session_state.training_logs.append(log_msg)
                        
                        # Auto-refresh
                        time.sleep(0.5)
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Training error: {str(e)}")
                    st.session_state.training_status = "error"
                    st.rerun()
            else:
                # Training completed
                st.session_state.training_status = "completed"
                
                # Save the model
                try:
                    model_path = trainer.save_model(f"final_{model_name}")
                    
                    # Add to trained models
                    if model_name not in st.session_state.trained_models:
                        st.session_state.trained_models.append({
                            'name': model_name,
                            'speaker': speaker_name,
                            'epochs': epochs,
                            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'model_path': model_path,
                            'final_loss': progress_info['current_loss']
                        })
                    
                    st.success("🎉 Real Training Completed Successfully!")
                    st.balloons()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error saving model: {str(e)}")
        
        elif st.session_state.training_status == "completed":
            st.success("🎉 Real Model Training Completed Successfully!")
            
            # Get the latest trained model
            if st.session_state.trained_models:
                latest_model = st.session_state.trained_models[-1]
                
                # Model info display
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**📁 Model Name:** {latest_model['name']}")
                    st.info(f"**🎤 Speaker:** {latest_model['speaker']}")
                with col2:
                    st.info(f"**🔄 Epochs:** {latest_model['epochs']}")
                    st.info(f"**📊 Final Loss:** {latest_model.get('final_loss', 'N/A')}")
                
                # Real model file download
                st.subheader("📥 Download Real Trained Model")
                st.markdown("**โมเดลที่เทรนจริงแล้ว!**")
                
                # Download real model file if it exists
                if 'model_path' in latest_model and os.path.exists(latest_model['model_path']):
                    with open(latest_model['model_path'], 'rb') as f:
                        model_data = f.read()
                    
                    st.download_button(
                        "🗂️ Download Real Model (.pth)",
                        data=model_data,
                        file_name=f"{latest_model['name'].replace(' ', '_')}_trained.pth",
                        mime="application/octet-stream",
                        help="ไฟล์โมเดลที่เทรนจริงด้วย PyTorch"
                    )
                    
                    # Model info
                    st.info(f"📊 Model size: {len(model_data) / 1024 / 1024:.1f} MB")
                else:
                    st.warning("⚠️ Model file not found. Please retrain the model.")
                
                # Training summary
                with st.expander("📝 Training Summary"):
                    st.write(f"**Model Path:** {latest_model.get('model_path', 'N/A')}")
                    st.write(f"**Training Completed:** {latest_model['created']}")
                    st.write(f"**Final Loss:** {latest_model.get('final_loss', 'N/A')}")
                    
                    if st.session_state.training_logs:
                        st.write("**Training Logs:**")
                        for log in st.session_state.training_logs[-5:]:
                            st.text(log)
            
            if st.button("🔄 Train New Model"):
                st.session_state.training_status = "ready"
                st.session_state.rvc_trainer = None
                st.session_state.training_logs = []
                st.rerun()
        
        # Training tips
        with st.expander("💡 Training Tips"):
            st.markdown("""
            **เทคนิคการเทรนโมเดลที่ดี:**
            
            🎤 **คุณภาพเสียง:**
            - ใช้ไฟล์เสียงคุณภาพสูง (WAV หรือ FLAC)
            - หลีกเลี่ยงเสียงรบกวน (background noise)
            - ความยาวไฟล์แต่ละไฟล์ 3-10 วินาที
            
            📊 **ข้อมูลการเทรน:**
            - อย่างน้อย 10 นาที สำหรับผลลัพธ์พื้นฐาน
            - 30-60 นาที สำหรับคุณภาพดี
            - 2+ ชั่วโมง สำหรับคุณภาพสูงสุด
            
            ⚡ **การตั้งค่า:**
            - เริ่มต้นด้วย default settings
            - เพิ่ม epochs หากผลลัพธ์ยังไม่ดี
            - ลด learning rate หากการเทรนไม่เสถียร
            """)

with tab4:
    st.header("Processing History")
    
    if st.session_state.conversion_history:
        st.subheader("🕒 Recent Conversions")
        
        for i, entry in enumerate(reversed(st.session_state.conversion_history[-10:])):
            with st.expander(f"Conversion {len(st.session_state.conversion_history)-i} - {entry['timestamp']}"):
                st.write(f"**Model Used:** {entry['model']}")
                st.write("**Parameters:**")
                for param, value in entry['parameters'].items():
                    st.write(f"  - {param.replace('_', ' ').title()}: {value}")
        
        if st.button("🗑️ Clear History"):
            st.session_state.conversion_history = []
            st.success("History cleared!")
            st.rerun()
    else:
        st.info("No conversion history yet. Start by converting some audio!")
    
    # Processing statistics
    st.subheader("📊 Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Conversions", len(st.session_state.conversion_history))
    
    with col2:
        if st.session_state.conversion_history:
            most_used_model = max(set([entry['model'] for entry in st.session_state.conversion_history]), 
                                 key=[entry['model'] for entry in st.session_state.conversion_history].count)
            st.metric("Most Used Model", most_used_model.split()[0])
        else:
            st.metric("Most Used Model", "N/A")
    
    with col3:
        st.metric("Available Models", len(available_models))

with tab5:
    st.header("About RVC-GUI Clone")
    
    st.markdown("""
    ### 🎤 Retrieval-based Voice Conversion GUI
    
    This is a Streamlit clone of the popular RVC-GUI voice conversion interface. It demonstrates
    the core functionality and user experience of voice conversion applications with mocked
    AI processing capabilities.
    
    #### ✨ Key Features:
    - **Multi-format Audio Support**: Upload WAV, MP3, FLAC, and other common audio formats
    - **Voice Recording**: Record audio directly in the browser
    - **Voice Conversion**: Transform voices using RVC (Retrieval-based Voice Conversion) models
    - **Vocal Separation**: Separate vocals from instrumental tracks using UVR5 models
    - **Model Training**: Train custom voice models with your own datasets
    - **Model Management**: Upload and manage custom voice models
    - **Parameter Control**: Fine-tune conversion with pitch, strength, and formant controls
    - **Processing History**: Track your conversion activities
    
    #### 🔧 Technical Details:
    - Built with Streamlit for rapid prototyping
    - Mock processing simulates realistic AI processing times
    - Supports common audio formats and operations
    - Responsive design for different screen sizes
    
    #### 🚀 Original Inspiration:
    This clone is based on the RVC-GUI hosted on Hugging Face Spaces, which provides
    advanced voice conversion capabilities using state-of-the-art AI models.
    
    ---
    
    **Note**: This is a demonstration interface with mocked AI processing. 
    In a production environment, this would integrate with actual RVC and UVR5 models.
    """)
    
    # System info
    with st.expander("🖥️ System Information"):
        st.write(f"**Hardware Mode**: {hardware_option}")
        st.write(f"**Processing Quality**: {quality}")
        st.write(f"**Active Model**: {selected_model}")
        st.write(f"**Session Conversions**: {len(st.session_state.conversion_history)}")

# Footer
st.markdown("---")
st.markdown("🎵 **RVC-GUI Clone** - Voice Conversion Interface | Built with Streamlit")
