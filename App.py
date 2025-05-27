import os
from flask import Flask, request, render_template, send_file, send_from_directory
from pydub import AudioSegment
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
SEGMENT_FOLDER = "static/segments"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENT_FOLDER, exist_ok=True)

def normalize(audio):
    return audio.set_frame_rate(16000).set_channels(1).apply_gain(-audio.max_dBFS)

def audio_to_np(audio_segment):
    return np.array(audio_segment.get_array_of_samples()).astype(np.float32)

def fast_cosine_similarity(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return np.dot(a, b) / (a_norm * b_norm)

def ms_to_timestamp(ms):
    seconds = ms // 1000
    return f"{seconds // 60:02}:{seconds % 60:02}.{ms % 1000:03}"

def match_segments(full_audio, sample_audio, step_ms=50, similarity_threshold=0.70):
    full_np = audio_to_np(full_audio)
    sample_np = audio_to_np(sample_audio)
    sample_len = len(sample_np)
    matches = []

    step_samples = int((step_ms / 1000) * full_audio.frame_rate)
    for i in range(0, len(full_np) - sample_len, step_samples):
        segment_np = full_np[i:i + sample_len]
        similarity = fast_cosine_similarity(sample_np, segment_np)
        if similarity > similarity_threshold:
            start_ms = int((i / full_audio.frame_rate) * 1000)
            end_ms = start_ms + len(sample_audio)
            matches.append((start_ms, end_ms, similarity))
    return matches

@app.route("/", methods=["GET", "POST"])
def index():
    matches = []
    if request.method == "POST":
        full_file = request.files["full_mp3"]
        sample_file = request.files["sample_mp3"]
        threshold = float(request.form["similarity"]) / 100.0

        full_path = os.path.join(UPLOAD_FOLDER, "given.mp3")
        sample_path = os.path.join(UPLOAD_FOLDER, "sample.mp3")
        full_file.save(full_path)
        sample_file.save(sample_path)

        full_audio = normalize(AudioSegment.from_file(full_path))
        sample_audio = normalize(AudioSegment.from_file(sample_path))

        matches = match_segments(full_audio, sample_audio, similarity_threshold=threshold)

        # Clear old segments
        for f in os.listdir(SEGMENT_FOLDER):
            os.remove(os.path.join(SEGMENT_FOLDER, f))

        # Export matched segments
        for idx, (start, end, sim) in enumerate(matches):
            clip = full_audio[start:end]
            filename = f"match_{idx+1}.mp3"
            clip.export(os.path.join(SEGMENT_FOLDER, filename), format="mp3")
            matches[idx] = {
                "start": ms_to_timestamp(start),
                "end": ms_to_timestamp(end),
                "similarity": f"{sim:.2f}",
                "filename": filename,
            }

    return render_template("index.html", matches=matches)

@app.route("/segments/<path:filename>")
def serve_segment(filename):
    return send_from_directory(SEGMENT_FOLDER, filename)


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host='0.0.0.0', port=port)