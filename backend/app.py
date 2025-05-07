from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from pyannote.audio import Pipeline
import pandas as pd
import speech_recognition as sr
import regex as re 

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 50MB
UPLOAD_FOLDER = 'uploads'
VOICE_FOLDER='voices'
UPLOAD_USER='users'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
result_df = pd.DataFrame(columns=["fileId", "speaker", "utterance","translation"])
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token="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"
)
selected_language = "en-IN"
summary=False

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_chunk(file, audio_file_path):
    print(f"Processing {file} for diarization...")
    diarization = pipeline(file)
    with open("audio.rttm", "w") as rttm:
        diarization.write_rttm(rttm)
    print(f"Overwritten audio.rttm with {file}")
    process_rttm_and_transcribe("audio.rttm", audio_file_path)

def process_rttm_and_transcribe(rttm_file_path, audio_file_path):
    global result_df,summary
    df = rttm_to_dataframe(rttm_file_path)
    df = df.astype({'start time': 'float', 'duration': 'float'})
    df['end time'] = df['start time'] + df['duration']

    df['utterance'] = df.apply(
        lambda row: extract_text_from_audio(audio_file_path, row['start time'], row['end time']),
        axis=1
    )

    grouped = []
    prev_speaker = None
    current_text = ""

    for _, row in df.iterrows():
        if not row['utterance']:
            continue
        if row['speaker'] == prev_speaker:
            current_text += " " + row['utterance']
        else:
            if prev_speaker is not None and current_text:
                grouped.append((audio_file_path, prev_speaker, current_text.strip()))
            prev_speaker = row['speaker']
            current_text = row['utterance']

    if prev_speaker and current_text:
        grouped.append((audio_file_path, prev_speaker, current_text.strip()))

    result_df = pd.DataFrame(grouped, columns=["fileId", "speaker", "utterance"])
    summary=True
    print(result_df)

def rttm_to_dataframe(rttm_file_path):
    columns = ["type", "fileId", "channel", "start time", "duration", "orthology", "confidence", "speaker", 'x', 'y']
    with open(rttm_file_path, "r") as rttm_file:
        lines = rttm_file.readlines()
        data = [line.strip().split() for line in lines]
        df = pd.DataFrame(data, columns=columns)
        df = df.drop(['x', 'y', "orthology", "confidence", "type", "channel"], axis=1)
        return df
    
def extract_text_from_audio(audio_file_path, start_time, end_time):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
        start_ms = int(start_time * 1000)
        end_ms = int((end_time + 0.2) * 1000)
        segment = audio.get_segment(start_ms, end_ms)
        try:
            return recognizer.recognize_google(segment, language=selected_language)
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            return f"API error: {e}"

@app.route('/')
def index():
    # Check if a file was uploaded (simple flag via query param)
    uploaded = request.args.get('uploaded')
    return render_template('index.html', uploaded=uploaded)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    return jsonify({'success': True}), 200


from flask import jsonify

@app.route('/process', methods=['POST'])
def process():
    filename = "uploads/Sample1.wav"
    process_chunk(filename, filename)
    return result_df.to_json(orient="records"), 200




@app.route('/clear', methods=['POST'])
def clear():
    print("clear was clicked")

    # Path to the uploads folder
    upload_folder = app.config['UPLOAD_FOLDER']

    # List all files in the uploads folder
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)

        # Check if it's a .wav file and remove it
        if filename.endswith('.wav'):
            try:
                os.remove(file_path)  # Delete the file
                print(f"Deleted {filename}")
            except Exception as e:
                print(f"Error deleting file {filename}: {e}")

    return "Files cleared", 200

@app.route('/uploadchunk', methods=['POST'])
def upload_audio():
    print("Received upload request")
    if 'audio' not in request.files:
        print("No audio file in request")
        return jsonify({'error': 'No audio file'}), 400

    file = request.files['audio']
    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(VOICE_FOLDER, filename)
    print(f"Saving to {filepath}")
    file.save(filepath)

    return jsonify({'success': True, 'filename': filename}), 200

@app.route('/transcribeUser')
def transcribe():
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile("users/newuser.wav") as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

            # Extract name and place using regex
            name_match = re.search(r'(?:hello|hi)[^a-zA-Z]+my name is ([A-Z][a-z]+(?: [A-Z][a-z]+)*)', text, re.IGNORECASE)
            place_match = re.search(r'I am from ([A-Z][a-z]+(?: [A-Z][a-z]+)*)', text, re.IGNORECASE)
            name = name_match.group(1) if name_match else ""
            place = place_match.group(1) if place_match else ""

            return jsonify({'success': True, 'text': text, 'name': name, 'place': place})

    except sr.UnknownValueError:
        return jsonify({'success': False, 'error': 'Could not understand audio'})
    except sr.RequestError as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/upload-User', methods=['POST'])
def upload_user():
    if 'audio' not in request.files:
        return jsonify({'message': '❌ No file uploaded'}), 400

    audio = request.files['audio']
    save_path = os.path.join(UPLOAD_USER, 'newuser.wav')
    audio.save(save_path)
    return jsonify({'message': '✅ Audio saved as newuser.wav'})


if __name__ == '__main__':
    app.run(debug=True)
