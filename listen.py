# Importing necessary libraries
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import soundfile as sf

# Load the model and processor
processor = AutoProcessor.from_pretrained("openai/whisper-small")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")

# Function to transcribe audio
def transcribe_audio(file_path):
    # Read the audio file
    audio_input, _ = sf.read(file_path)
    
    # Process the audio input
    inputs = processor(audio_input, sampling_rate=16_000, return_tensors="pt")
    
    # Generate the transcription
    with torch.no_grad():
        logits = model(**inputs).logits

    # Decode the predicted ids to text
    transcription = processor.batch_decode(logits.numpy()).text

    return transcription

# Transcribe the audio file
transcription = transcribe_audio("audio.wav")

# Print the transcription
print("Transcription:\n", transcription)
