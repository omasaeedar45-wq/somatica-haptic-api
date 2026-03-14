from flask import Flask, request, jsonify
import librosa
import numpy as np
import tempfile
import os
import requests

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "somatica-haptic-api"})

@app.route('/convert', methods=['POST'])
def convert():
    data = request.get_json()
    
    if not data or 'audio_url' not in data:
        return jsonify({"error": "audio_url is required"}), 400

    audio_url = data['audio_url']
    
    try:
        # Download the audio file from Supabase
        response = requests.get(audio_url, timeout=30)
        response.raise_for_status()
        
        # Save to temp file
        suffix = '.mp3' if 'mp3' in audio_url.lower() else '.wav'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # Load audio with librosa
        y, sr = librosa.load(tmp_path, sr=22050, mono=True)
        os.unlink(tmp_path)

        # Get duration in seconds
        duration = float(librosa.get_duration(y=y, sr=sr))

        # Analyze amplitude envelope (how loud each moment is)
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_normalized = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9)

        # Detect beats and transients (taps,​​​​​​​​​​​​​​​​
