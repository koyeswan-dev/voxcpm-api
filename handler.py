import runpod
from voxcpm import VoxCPM
import soundfile as sf
import base64
import io
import os
import tempfile
import urllib.request
import subprocess

# Bypass Catbox anti-bot protection (Catbox ကို လှည့်စားခြင်း)
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

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
        # Download MP3 and convert to WAV
        if ref_audio_url and ref_audio_url.startswith('http'):
            temp_mp3 = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            
            # Close file handles before passing to external tools
            temp_mp3.close()
            temp_wav.close()
            
            # Download using the spoofed opener
            urllib.request.urlretrieve(ref_audio_url, temp_mp3.name)
            
            # Convert MP3 to WAV seamlessly
            subprocess.run(["ffmpeg", "-y", "-i", temp_mp3.name, "-ar", "16000", temp_wav.name], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            ref_wav_path = temp_wav.name
            print(f"Reference audio ready: {ref_wav_path}")
        
        # Generate audio using reference voice
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
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return {"error": str(e)}
        
    finally:
        # Clean up memory securely
        if temp_mp3 and os.path.exists(temp_mp3.name):
            os.unlink(temp_mp3.name)
        if temp_wav and os.path.exists(temp_wav.name):
            os.unlink(temp_wav.name)

runpod.serverless.start({"handler": handler})
