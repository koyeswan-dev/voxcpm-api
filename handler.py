import runpod
from voxcpm import VoxCPM
import soundfile as sf
import base64
import io

print("Loading model...")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2")
print("Model loaded.")

def handler(job):
    job_input = job.get('input', {})
    text = job_input.get('text', 'Hello')
    wav = model.generate(text=text)
    buffer = io.BytesIO()
    sf.write(buffer, wav, model.tts_model.sample_rate, format='WAV')
    audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return {"audio_base64": audio_base64}

runpod.serverless.start({"handler": handler})
