import openai
from pathlib import Path
import os
from dotenv import load_dotenv
import pyaudio
import wave
import tempfile

load_dotenv()

class VoiceController:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.audio = pyaudio.PyAudio()
        
    def listen_for_command(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        RECORD_SECONDS = 20
        
        stream = self.audio.open(format=FORMAT,
                               channels=CHANNELS,
                               rate=RATE,
                               input=True,
                               frames_per_buffer=CHUNK)
        
        print("Listening for command...")
        frames = []
        
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            
        stream.stop_stream()
        stream.close()
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                wf = wave.open(temp_audio.name, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                temp_audio_path = temp_audio.name

            with open(temp_audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            os.unlink(temp_audio_path)
            
            text = transcript.text
            return text
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None