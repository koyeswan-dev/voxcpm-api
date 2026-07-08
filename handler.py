import runpod
from voxcpm import VoxCPM
import soundfile as sf
import base64
import io
import os
import tempfile
import urllib.request
import subprocess

print("Loading model...")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2")
print("Model loaded.")

def handler(job):
    job_input = job.get('input', {})
    text = job_input.get('text', 'Hello')
    ref_audio_url = job_input.get('ref_audio', None)
    
    ref_wav_path = None
    temp_mp3 = None
    temp_wav = None
    
    try:
        # Download and convert to WAV
        if ref_audio_url and ref_audio_url.startswith('http'):
            temp_mp3 = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            
            urllib.request.urlretrieve(ref_audio_url, temp_mp3.name)
            # Convert MP3 to WAV using FFmpeg
            subprocess.run(["ffmpeg", "-y", "-i", temp_mp3.name, "-ar", "16000", temp_wav.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            ref_wav_path = temp_wav.name
            print(f"Reference audio ready: {ref_wav_path}")
        
        # Generate with or without reference
        if ref_wav_path:
            wav = model.generate(
                text=text,
                reference_wav_path=ref_wav_path,
                cfg_value=2.0
            )
        else:
            wav = model.generate(text=text)
        
        buffer = io.BytesIO()
        sf.write(buffer, wav, model.tts_model.sample_rate, format='WAV')
        audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return {"audio_base64": audio_base64}
    
    finally:
        # Cleanup temp files
        if temp_mp3 and os.path.exists(temp_mp3.name):
            os.unlink(temp_mp3.name)
        if temp_wav and os.path.exists(temp_wav.name):
            os.unlink(temp_wav.name)

runpod.serverless.start({"handler": handler})
