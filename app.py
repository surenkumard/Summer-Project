import os
from flask import Flask, render_template, request, send_from_directory, jsonify
from pydub import AudioSegment
import speech_recognition as sr
from werkzeug.utils import secure_filename
import re
from pytube import YouTube
from yt_dlp import YoutubeDL
from googletrans import Translator
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav', 'ogg'}

g_chunks = []  # Global variable to store chunks
g_lang = ""
g_full_text = ""

# Initialize the translation and summarization tools
translator = Translator()
model_name = "sshleifer/distilbart-cnn-12-6"
summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)

def chunk_text(text, tokenizer, max_length=1024):
    # Use the tokenizer to encode the text into tokens
    sentences = tokenizer.encode(text, return_tensors='pt', truncation=True)
    sentences = tokenizer.decode(sentences[0]).split('.')  # Split sentences based on period
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Clean up sentence
        sentence = sentence.strip()
        if not sentence:  # Skip empty sentences
            continue
        
        if len(tokenizer(current_chunk + " " + sentence, return_tensors='pt')['input_ids'][0]) < max_length:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def summarize_text(text, max_length=1024):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    text_chunks = chunk_text(text, tokenizer, max_length)
    
    summaries = []
    for chunk in text_chunks:
        input_length = len(chunk.split())  # Calculate the word count of the chunk

        # Calculate min_length as 30% and max_length as 50% of the input text length
        min_length = int(0.3 * input_length)
        max_length = int(0.5 * input_length)
        
        # Ensure there's a reasonable minimum length
        if min_length < 25:
            min_length = 25
        if max_length < 50:
            max_length = 50

        # Generate summary based on percentage-based length
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    final_summary = " ".join(summaries)

    return final_summary

    
@app.route('/final_summarizer', methods=['GET', 'POST'])
def final_summarizer():
    global g_full_text
    if g_lang == 'en':
        summarized_text = summarize_text(g_full_text)
        
        return {'full_summary' : summarized_text}
    elif g_lang == 'ta-IN':
        
    # Translate Tamil text to English
        translated_text = translate_to_english(g_full_text)

        if translated_text:
        # Summarize the translated English text
            summarized_text = summarize_text(translated_text)

        # Translate the summary back to Tamil
            final_summary_tamil = translate_to_tamil(summarized_text)

            return {'full_summary' : final_summary_tamil}
        else:
            return {'error' : 'failed to summarize Tamil content'}
    else:
        return {'error' : 'Mixed Language. '}

# Check if file type is allowed
def allowed_file(l_filename):
    return '.' in l_filename and l_filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to split audio into chunks
def split_audio_into_chunks(l_wav_path, l_chunk_length_ms):
    """Split audio into chunks of specified length."""
    global g_full_text, g_chunks
    l_audio = AudioSegment.from_file(l_wav_path)
    l_chunks = []
    g_chunks = []
    g_full_text = ""
    for i in range(0, len(l_audio), l_chunk_length_ms):
        l_chunk = l_audio[i:i + l_chunk_length_ms]
        l_chunk_path = f"{l_wav_path}_chunk_{i}.wav"
        l_chunk.export(l_chunk_path, format="wav")
        l_chunks.append(l_chunk_path)
    return l_chunks

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        global g_chunks  # To store the chunks globally
        if 'file' not in request.files or not request.form.get('language'):
            return 'Please upload a file and select a language.', 400

        l_file = request.files['file']
        global g_lang
        g_lang = "en"

        if l_file.filename == '':
            return 'No selected file.', 400

        if l_file and allowed_file(l_file.filename):
            l_filename = l_file.filename
            l_file_path = os.path.join(app.config['UPLOAD_FOLDER'], l_filename)
            l_file.save(l_file_path)

            # Convert to .wav format and split into chunks
            l_audio = AudioSegment.from_mp3(l_file_path)
            l_wav_path = os.path.splitext(l_file_path)[0] + ".wav"
            l_audio.export(l_wav_path, format="wav")

            g_chunks = split_audio_into_chunks(l_wav_path, l_chunk_length_ms=60000)
            return {'audio_file': l_filename, 'length': len(g_chunks)}, 200
    
    return render_template('files.html')

@app.route('/transcribe_chunk/<int:chunk_index>', methods=['GET'])
def transcribe_chunk(chunk_index):
    global g_chunks, g_full_text
    if 0 <= chunk_index < len(g_chunks):
        l_chunk_path = g_chunks[chunk_index]
        r = sr.Recognizer()
        with sr.AudioFile(l_chunk_path) as source:
            l_audio = r.record(source)
            try:
                l_text = r.recognize_google(l_audio, language=g_lang)
                g_full_text += l_text
                return l_text
            except sr.UnknownValueError:
                return "Unable to transcribe audio in chunk."
            except sr.RequestError as e:
                return f"Error in chunk: {e}"
    else:
        return 'Invalid chunk index'
    
def normalize_filename(filename):
    """Normalize filename by removing special characters and whitespace."""
    normalized = re.sub(r'[^\w\s-]', '', filename).strip().lower()
    normalized = re.sub(r'[-\s]+', '_', normalized)
    return normalized

def process_video_url(url):
    """Download video from URL and extract audio."""
    try:
        # Define yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(app.config['UPLOAD_FOLDER'], '%(title).3s.%(ext)s'),
            'noplaylist': True,
        }

        # Use yt-dlp to download the video
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info_dict)
        
        # Normalize filename
        normalized_filename = normalize_filename(info_dict['title'][:3]) + ".mp3"

        return normalized_filename

    except Exception as e:
        raise Exception(f'Failed to process video from URL: {str(e)}')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def extract_youtube_id(url):
    """Extract the YouTube video ID from the URL."""
    regex = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(regex, url)
    return match.group(1) if match else None

@app.route('/url', methods=['GET', 'POST'])
def url_input():
    if request.method == 'POST':
        global g_chunks, g_lang
        g_lang = 'en'
        g_chunks = []
        
        video_url = request.form.get('videoUrl')
        video_id = extract_youtube_id(video_url)
        
        if video_id:
            embed_url = f"https://www.youtube.com/embed/{video_id}"
            
             # Process video and extract audio
            audio_file_name = process_video_url(video_url)  # Process the video URL
            audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file_name)
            if not audio_file_name:
                return jsonify({'error': 'Failed to process video URL'}), 500

            # Convert audio to wav and split into chunks
            l_audio = AudioSegment.from_mp3(audio_file_path)
            l_wav_path = os.path.splitext(audio_file_path)[0] + ".wav"
            l_audio.export(l_wav_path, format="wav")

            g_chunks = split_audio_into_chunks(l_wav_path, l_chunk_length_ms=60000)
            
            return {'embed_url': embed_url, 'length': len(g_chunks)}, 200
        else:
            error = "Invalid YouTube URL. Please try again."
            return render_template('url.html', error=error)
    return render_template('url.html')

@app.route('/download_transcription', methods=['POST'])
def download_transcription():
    text = request.form.get('transcription_text')
    if text is None:
        return jsonify(error="No transcription text provided"), 400

    filename = 'transcription.txt'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(file_path, 'w') as file:
        file.write(text)

    return send_from_directory(directory=app.config['UPLOAD_FOLDER'], path=filename, as_attachment=True)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)