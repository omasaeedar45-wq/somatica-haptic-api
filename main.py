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
    tmp_path = None

    try:
        r = requests.get(audio_url, timeout=30)
        r.raise_for_status()
        suffix = '.mp3' if 'mp3' in audio_url.lower() else '.wav'
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(r.content)
        tmp.close()
        tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=22050, mono=True)
        duration = float(librosa.get_duration(y=y, sr=sr))
        hop_length = 512

        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_min = rms.min()
        rms_max = rms.max()
        rms_norm = (rms - rms_min) / (rms_max - rms_min + 1e-9)

        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

        haptic_events = []

        for onset_time in onset_times:
            frame_idx = librosa.time_to_frames(onset_time, sr=sr, hop_length=hop_length)
            frame_idx = min(int(frame_idx), len(rms_norm) - 1)
            intensity = float(rms_norm[frame_idx])
            if intensity > 0.08:
                haptic_events.append({
                    "time": round(float(onset_time), 3),
                    "intensity": round(intensity, 3),
                    "duration": int(80 + intensity * 220)
                })

        t = 0.0
        while t < duration:
            frame_idx = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
            frame_idx = min(int(frame_idx), len(rms_norm) - 1)
            intensity = float(rms_norm[frame_idx]) * 0.4
            if intensity > 0.05:
                haptic_events.append({
                    "time": round(t, 3),
                    "intensity": round(intensity, 3),
                    "duration": 180
                })
            t += 2.0

        haptic_events.sort(key=lambda x: x["time"])

        return jsonify({
            "duration": round(duration, 2),
            "total_events": len(haptic_events),
            "hapticEvents": haptic_events
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
