import runpod
from voxcpm import VoxCPM
import soundfile as sf
import base64
import io
import os
import tempfile
import urllib.request

print("Loading model...")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2")
print("Model loaded.")

def handler(job):
    job_input = job.get('input', {})
    text = job_input.get('text', 'Hello')
    ref_audio_url = job_input.get('ref_audio', None)
    
    ref_wav_path = None
    temp_file = None
    
    try:
        # Download reference audio if URL is provided
        if ref_audio_url and ref_audio_url.startswith('http'):
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            urllib.request.urlretrieve(ref_audio_url, temp_file.name)
            ref_wav_path = temp_file.name
            print(f"Reference audio downloaded: {ref_wav_path}")
        
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
        # Cleanup temp file
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

runpod.serverless.start({"handler": handler})
