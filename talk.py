import json
import random
import sounddevice as sd
import scipy.io.wavfile
from transformers import AutoProcessor, AutoModel
import pyaudio
import wave
import whisper
import warnings
import subprocess

# Function to record audio
def record_audio(filename, duration=5):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)

    print("Recording...")

    frames = []

    for i in range(0, int(16000 / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("Finished recording.")

    # Stop the stream and close PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data to a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def generate_speech(text, processor, model):
    inputs = processor(text=[text], voice_preset="v2/en_speaker_9", return_tensors="pt")
    speech_values = model.generate(**inputs, do_sample=True)
    audio_data = speech_values.cpu().numpy().squeeze()
    first_word = text.split(" ")[0]
    random_number = random.randint(1000, 9999)
    filename = f"{first_word}{random_number}.wav"
    scipy.io.wavfile.write(filename, rate=24000, data=audio_data)
    sd.play(audio_data, samplerate=24000)
    sd.wait()
    return text, filename

def process_text(text_buffer, processor, model, corpus):
    text, filename = generate_speech(text_buffer, processor, model)
    corpus.append({"text": text, "filename": filename})
    with open("corpus.json", "w") as f:
        json.dump(corpus, f)

def generate_text(prompt, processor, model, corpus, text_model_url="http://localhost:11434/api/generate"):
    cmd = [
        'curl',
        '--silent',
        '--show-error',
        '-X', 'POST',
        text_model_url,
        '-d',
        json.dumps({"model": "mistral", "prompt": prompt})
    ]
    text_buffer = ''
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in iter(process.stdout.readline, ''):
        line = line.decode('utf-8').strip()
        if not line:
            continue
        parsed_json = json.loads(line)
        response_token = parsed_json.get('response', '')
        text_buffer += response_token
        if text_buffer and text_buffer[-1] in ['.','\n']:
            print(f"Buffered Text: {text_buffer}")
            process_text(text_buffer, processor, model, corpus)
            text_buffer = ''
    if text_buffer:
        print(f"Buffered Text at EOF: {text_buffer}")
        process_text(text_buffer, processor, model, corpus)

def main():
    warnings.filterwarnings('ignore')
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = AutoModel.from_pretrained("suno/bark-small")
    try:
        with open("corpus.json", "r") as f:
            corpus = json.load(f)
        print("Loaded existing corpus:", corpus)
    except FileNotFoundError:
        corpus = []

    record_audio('audio.wav')

    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("audio.wav")
    transcription = result.get('text', '')
    print(f'Transcription: {transcription}')


    generate_text(transcription, processor, model, corpus)

if __name__ == "__main__":
    main()
