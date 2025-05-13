from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from pyannote.audio import Pipeline,Model,Inference
import pandas as pd
import speech_recognition as sr
import regex as re 
import openai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist
from pydub import AudioSegment
import glob
import shutil
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from IndicTransToolkit import IndicProcessor
import torch

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 50MB
UPLOAD_FOLDER = 'uploads'
VOICE_FOLDER='voices'
UPLOAD_USER='users'
SEGMENT_DIR = "segments"
segment_counter = 1
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
result_df = pd.DataFrame(columns=["fileId", "speaker", "utterance","translation"])
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token="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"
)
embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token="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")
inference = Inference(embedding_model, window="whole")
# Speaker label storage
speaker_embeddings = []
segment_speakers = []
speaker_names = []
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
selected_language = "en-IN"
summary=False
executor = ThreadPoolExecutor(max_workers=5) 
THRESHOLD = 0.8 
transcript_lines = []
unknown_speaker_count = 1
last_speaker = None

#indic
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None


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
    # Save the DataFrame to transcript.txt in a readable format
    with open("transcript.txt", "w", encoding="utf-8") as f:
        for _, row in result_df.iterrows():
            f.write(f"{row['speaker']}: {row['utterance']}\n")


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
def diarize_and_segment(chunk_path, rttm_path):
    global segment_counter, speaker_embeddings, segment_speakers, last_speaker, transcript_lines

    print(f" Diarizing {chunk_path}...")
    diarization = pipeline(chunk_path)
    print("Diarization result:", diarization)
    with open(rttm_path, "w") as f:
        diarization.write_rttm(f)

    print(f" RTTM saved to {rttm_path}")
    audio = AudioSegment.from_wav(chunk_path)

    df = pd.read_csv(rttm_path, sep=" ", header=None, comment=";", names=[
        "Type", "File ID", "Channel", "Start", "Duration",
        "NA1", "NA2", "Speaker", "NA3", "NA4"
    ])

    for _, row in df.iterrows():
        start = row["Start"]
        duration = row["Duration"]

        if duration < 0.5:
            continue

        buffer=800
        end = start + duration
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)+buffer 

        segment_audio = audio[start_ms:end_ms]
        segment_filename = f"segment_{segment_counter}.wav"
        segment_path = os.path.join(SEGMENT_DIR, segment_filename)
        segment_audio.export(segment_path, format="wav")
        print(f" Saved: {segment_filename} | Start={start:.3f}s Duration={duration:.3f}s")
        segment_counter += 1

        # Transcribe the segment
        transcript = extract_text_from_audio(segment_path, start_time=0, end_time=duration)
        if transcript.strip() == "":
            print(f" Skipping {segment_filename} — empty transcription")
            continue  # Skip embedding and labeling

        embeddings_root = "embeddings"
        wav_files = glob.glob(os.path.join(embeddings_root, "*.wav"))

        # Regex to extract speaker name from filename (e.g., nitik2.wav → nitik)
        speaker_pattern = re.compile(r"([a-zA-Z]+)")

        for wav_path in wav_files:
            filename = os.path.basename(wav_path)
            match = speaker_pattern.match(filename)
            if match:
                speaker_name = match.group(1)
                embedding = inference(wav_path).reshape(1, -1)
                speaker_embeddings.append(embedding)
                speaker_names.append(speaker_name)

        emb = inference(segment_path).reshape(1, -1)
        print(f'current={emb}')
        if not speaker_embeddings:
            speaker_embeddings.append(emb)
            speaker_label = "Speaker_1"
            print(f" {segment_filename} → {speaker_label} (first speaker)")
        else:
            distances = [cdist(emb, known_emb, metric="cosine")[0, 0] for known_emb in speaker_embeddings]
            min_dist = min(distances)

            if min_dist <= THRESHOLD:
                speaker_idx = distances.index(min_dist)
                speaker_label = speaker_names[speaker_idx]
            else:
                new_label = f"unknown_speaker_{unknown_speaker_count}"
                unknown_speaker_count += 1  # Increment for next unknown speaker
                speaker_embeddings.append(emb)
                speaker_names.append(new_label)
                speaker_label = new_label

            print(f"{segment_filename} → {speaker_label} (min_dist={min_dist:.4f})")
            segment_speakers.append((segment_filename, speaker_label))
            print(f"{speaker_label}: {transcript}")

        # Merge if same speaker as previous
        if speaker_label == last_speaker and transcript_lines:
            transcript_lines[-1] = transcript_lines[-1].strip() + f" {transcript}"
        else:
            transcript_lines.append(f"{speaker_label}: {transcript}")
        last_speaker = speaker_label

        # Write the updated transcript to file
        t_file="live.txt"
        with open(t_file, "w", encoding="utf-8") as f:
            f.write("\n".join(transcript_lines))

def getTranslation(content):
    global selected_language
    def initialize_model_and_tokenizer(ckpt_dir, quantization):
        if quantization == "4-bit":
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8-bit":
            qconfig = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
        else:
            qconfig = None

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=qconfig,
        )

        if qconfig == None:
            model = model.to(DEVICE)
            if DEVICE == "cuda":
                model.half()

        model.eval()

        return tokenizer, model


    def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
        translations = []
        for i in range(0, len(input_sentences), BATCH_SIZE):
            batch = input_sentences[i : i + BATCH_SIZE]

            # Preprocess the batch and extract entity mappings
            batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

            # Tokenize the batch and generate input encodings
            inputs = tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(DEVICE)

            # Generate translations using the model
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )

            # Decode the generated tokens into text

            with tokenizer.as_target_tokenizer():
                generated_tokens = tokenizer.batch_decode(
                    generated_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

            # Postprocess the translations, including entity replacement
            translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

            del inputs
            torch.cuda.empty_cache()

        return translations

    indic_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-1B"  # ai4bharat/indictrans2-indic-en-dist-200M
    indic_en_tokenizer, indic_en_model = initialize_model_and_tokenizer(indic_en_ckpt_dir, quantization)

    ip = IndicProcessor(inference=True)
    if selected_language=="hi-IN":
        src_lang, tgt_lang = "hin_Deva", "eng_Latn"
    elif selected_language=="ta-IN":
        src_lang, tgt_lang = "tam_Taml", "eng_Latn"
    elif selected_language=="te-IN":
        src_lang, tgt_lang = "tel_Telu", "eng_Latn"
    elif selected_language=="bn-IN":
        src_lang, tgt_lang = "ben_Beng", "eng_Latn"
    elif selected_language=="gu-IN":
        src_lang, tgt_lang = "guj_Gujr", "eng_Latn"
    elif selected_language=="kn-IN":
        src_lang, tgt_lang = "kan_Knda", "eng_Latn"
    elif selected_language=="ml-IN":
        src_lang, tgt_lang = "mal_Mlym", "eng_Latn"
    elif selected_language=="mr-IN":
        src_lang, tgt_lang = "mar_Deva", "eng_Latn"
    elif selected_language=="pa-IN":
        src_lang, tgt_lang = "pan_Guru", "eng_Latn"
    elif selected_language=="ur-IN":
        src_lang, tgt_lang = "urd_Arab", "eng_Latn"
    en_translations = batch_translate(content, src_lang, tgt_lang, indic_en_model, indic_en_tokenizer, ip)
    

    print(f"\n{src_lang} - {tgt_lang}")
    for input_sentence, translation in zip(content, en_translations):
        print(f"{src_lang}: {input_sentence}")
        print(f"{tgt_lang}: {translation}")
        
    
    return en_translations
    

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
    wav_files = glob.glob("uploads/*.wav")

    if wav_files:
        filename = wav_files[0]  # Since there's only one file
        print(f"Found file: {filename}")
    else:
        print("No .wav file found.")
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
    if os.path.exists("transcript.txt"):
        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.truncate(0)
        print("transcript.txt truncated (emptied).")    
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

    # Run diarization and segmentation
    rttm_path = filepath.replace(".wav", ".rttm")
    executor.submit(diarize_and_segment, filepath, rttm_path)

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

@app.route('/get_summary', methods=['GET'])
def get_summary():
    global summary_ready
    transcript_path = "transcript.txt"
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    else:
        full_text = ""

    if full_text.strip():
        prompt = f"""
        You are a helpful assistant. Please read the following meeting transcript and return the following:

        1. A complete summary of the conversation 
        2. Key discussion points (as bullet points)
        3. Action items (as bullet points)

        Transcript:
        {full_text}

        Format your response as:
        Summary: ...
        Key Points:
        - ...
        Actions:
        - ...
        """
        try:
            import openai
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            content = response['choices'][0]['message']['content']
            summary_part = content.split("Key Points:")[0].replace("Summary:", "").strip()
            keypoints_part = content.split("Key Points:")[1].split("Actions:")[0].strip()
            actions_part = content.split("Actions:")[1].strip()

            app.config["SUMMARY"] = {
                "summary": summary_part,
                "key_points": keypoints_part,
                "actions": actions_part
            }
            summary_ready = True
            print("✅ Summary generation complete.")
        except Exception as e:
            print(f"❌ Error during summary generation: {e}")
            app.config["SUMMARY"] = {"summary": "", "key_points": "", "actions": ""}
            summary_ready = True
    else:
        print("⚠️ Empty transcript. No summary generated.")
        app.config["SUMMARY"] = {"summary": "", "key_points": "", "actions": ""}
        summary_ready = True
    
        

    return jsonify(app.config.get("SUMMARY", {
        "summary": "",
        "key_points": "",
        "actions": ""
    }))



@app.route('/set_language', methods=['POST'])
def set_language():
    global selected_language
    data = request.get_json()
    selected_language = data.get('language', 'en-IN')
    print("🔤 Language set to:", selected_language)
    return jsonify({'success': True})

@app.route('/get_transcript')
def get_transcript():
    transcript_path = "live.txt" 
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Filter out lines where the content is None or empty after colon
        cleaned_lines = [line.strip() for line in lines if ":" in line and line.strip().split(":", 1)[1].strip().lower() not in ["", "none"]]
        return jsonify({"transcript": "\n\n".join(cleaned_lines)})
    return jsonify({"transcript": ""})

@app.route('/get_summary_live', methods=['GET'])
def get_summary_live():
    global summary_ready
    transcript_path = "live.txt"
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    else:
        full_text = ""

    if full_text.strip():
        prompt = f"""
        You are a helpful assistant. Please read the following meeting transcript and return the following:

        1. A complete summary of the conversation without missing any points
        2. Key discussion points (as bullet points) with which speaker said what
        3. Action items (as bullet points) which speaker should do what

        Transcript:
        {full_text}

        Format your response as:
        Summary: ...
        Key Points:
        - ...
        Actions:
        - ...
        """
        try:
            import openai
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            content = response['choices'][0]['message']['content']
            summary_part = content.split("Key Points:")[0].replace("Summary:", "").strip()
            keypoints_part = content.split("Key Points:")[1].split("Actions:")[0].strip()
            actions_part = content.split("Actions:")[1].strip()

            app.config["SUMMARY"] = {
                "summary": summary_part,
                "key_points": keypoints_part,
                "actions": actions_part
            }
            summary_ready = True
            print("✅ Summary generation complete.")
        except Exception as e:
            print(f"❌ Error during summary generation: {e}")
            app.config["SUMMARY"] = {"summary": "", "key_points": "", "actions": ""}
            summary_ready = True
    else:
        print("⚠️ Empty transcript. No summary generated.")
        app.config["SUMMARY"] = {"summary": "", "key_points": "", "actions": ""}
        summary_ready = True
    
        

    return jsonify(app.config.get("SUMMARY", {
        "summary": "",
        "key_points": "",
        "actions": ""
    }))

@app.route('/clear_live', methods=['POST'])
def clearLive():
    global segment_counter
    for folder in [VOICE_FOLDER, SEGMENT_DIR]:
        if os.path.exists(folder):
            print(f"Cleaning up {folder}: {os.listdir(folder)}")
            for f in os.listdir(folder):
                try:
                    os.remove(os.path.join(folder, f))
                except Exception as e:
                    print(f"Error deleting file {f}: {e}")
        else:
            print(f"Folder {folder} does not exist.")
    
    # Truncate live.txt instead of deleting it
    if os.path.exists("live.txt"):
        with open("live.txt", "w", encoding="utf-8") as f:
            f.truncate(0)
        print("live.txt truncated (emptied).")
    segment_counter=1
    return jsonify({'status': '✅ All chunks, segments, and transcript cleared'})

@app.route('/register-speaker', methods=['POST'])
def register_speaker():
    data = request.get_json()
    name = data.get('name')

    if not name:
        return jsonify({'success': False, 'error': 'Missing name'})

    source_path = os.path.join("users", "newuser.wav")
    dest_path = os.path.join("embeddings", f"{name}.wav")

    try:
        # Move and rename the audio file
        if not os.path.exists(source_path):
            return jsonify({'success': False, 'error': 'Recorded audio not found'})

        shutil.move(source_path, dest_path)

        # Generate embedding
        embedding = inference(dest_path).reshape(1, -1)
        speaker_embeddings.append(embedding)
        speaker_names.append(name)

        return jsonify({'success': True, 'message': 'Speaker registered'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/get-translation')
def get_translation():
    transcript_path = "live.txt"
    if not os.path.exists(transcript_path):
        return jsonify({'translation': "Transcript not found."})

    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()
        content=[content]
    translation = getTranslation(content)
    return jsonify({'translation': translation})

@app.route('/get-translation-file')
def get_translation_file():
    transcript_path = "transcript.txt"
    if not os.path.exists(transcript_path):
        return jsonify({'translation': "Transcript not found."})

    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()
        content=[content]
    translation = getTranslation(content)
    return jsonify({'translation': translation})


if __name__ == '__main__':
    app.run(debug=True)
