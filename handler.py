ref_wav_path = None
  temp_input = None
  temp_wav = None
  
  try:
      if ref_audio_base64:
          # Decode Base64 string sent from Flutter App
          audio_bytes = base64.b64decode(ref_audio_base64)
          temp_input = tempfile.NamedTemporaryFile(suffix='.tmp', delete=False)
          temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
          
          temp_input.write(audio_bytes)
          temp_input.close()
          temp_wav.close()
          
          # Convert to 16kHz WAV format required by VoxCPM
          subprocess.run(["ffmpeg", "-y", "-i", temp_input.name, "-ar", "16000", temp_wav.name], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
          ref_wav_path = temp_wav.name
          
      elif ref_audio_url and ref_audio_url.startswith('http'):
          # Fallback URL download logic (with Mozilla User-Agent to bypass blocks)
          req = urllib.request.Request(ref_audio_url, headers={'User-Agent': 'Mozilla/5.0'})
          with urllib.request.urlopen(req) as response:
              audio_data = response.read()
              
          temp_input = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
          temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
          temp_input.write(audio_data)
          temp_input.close()
          temp_wav.close()
          
          subprocess.run(["ffmpeg", "-y", "-i", temp_input.name, "-ar", "16000", temp_wav.name], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
          ref_wav_path = temp_wav.name
      else:
          raise Exception("No valid reference audio provided. Please provide ref_audio or ref_audio_base64.")
  
      print(f"Generating audio for text: {text}")
      
      # Load and verify reference audio
      ref_waveform, _ = torchaudio.load(ref_wav_path)
      if ref_waveform.shape[0] > 1:
          ref_waveform = ref_waveform.mean(dim=0, keepdim=True)
          
      # Generate cloned audio
      generated_waveform = model.generate(text, ref_audio=ref_wav_path)
      
      output_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
      output_temp.close()
      
      # Save output to temp file
      torchaudio.save(output_temp.name, generated_waveform.cpu(), 24000)
      
      # Read output and convert to Base64 to return to Flutter
      with open(output_temp.name, 'rb') as f:
          out_bytes = f.read()
          out_base64 = base64.b64encode(out_bytes).decode('utf-8')
          
      return {"audio_base64": out_base64}
      
  except Exception as e:
      print(f"Error: {e}")
      return {"error": str(e)}
  finally:
      # Cleanup all temporary files to save disk space
      for path in [temp_input, temp_wav]:
          if path and os.path.exists(path.name):
              os.remove(path.name)
      if ref_wav_path and os.path.exists(ref_wav_path):
          os.remove(ref_wav_path)
      if 'output_temp' in locals() and os.path.exists(output_temp.name):
          os.remove(output_temp.name)
