## Package
import os
import json
import pyaudio
import wave
import time
import streamlit as st
from datetime import datetime
import tempfile
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.document_loaders import JSONLoader
import audio_processing.lyrics_recognition as lyrics_recognition
import audio_processing.pitch_detection as pitch_detection
import audio_processing.process_lyrics_note as process_lyrics_note
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.errors import InvalidDimensionException
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from typing import Dict, List, Any
from glob import glob
from gtts import gTTS
from midiutil import MIDIFile
import pygame
import io
from midi2audio import FluidSynth
import soundfile as sf

# Set page configuration
st.set_page_config(page_title="我不是胖虎", layout="wide")
st.title("我不是胖虎")

# Initialize session state variables if they don't exist
if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'feedback' not in st.session_state:
    st.session_state.feedback = None
if 'audio_response' not in st.session_state:
    st.session_state.audio_response = None
if 'ref_midi_path' not in st.session_state:
    st.session_state.ref_midi_path = None
if 'sing_midi_path' not in st.session_state:
    st.session_state.sing_midi_path = None
if 'transcript_path' not in st.session_state:
    st.session_state.transcript_path = None

# OpenAI API Key input
api_key = st.sidebar.text_input("輸入您的 OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.sidebar.success("API Key 已設置!")
else:
    st.sidebar.warning("請輸入 OpenAI API Key 以使用此應用")

## 錄音功能
def record_audio(output_path="./audio/recorded_audio.wav", record_seconds=10, sample_rate=44100, channels=1):
    """
    即時錄製音訊並儲存為檔案
    
    參數:
        output_path: 錄音檔案的儲存路徑
        record_seconds: 錄音時間長度(秒)
        sample_rate: 取樣率
        channels: 聲道數
    
    回傳:
        錄音檔案的路徑
    """
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 設定錄音參數
    chunk = 1024
    format = pyaudio.paInt16
    
    # 初始化 PyAudio
    p = pyaudio.PyAudio()
    
    st.info(f"開始錄音，將持續 {record_seconds} 秒...")
    
    # 開啟錄音串流
    stream = p.open(format=format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)
    
    frames = []
    
    # 錄音
    progress_bar = st.progress(0)
    for i in range(0, int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)
        # 更新進度條
        progress = i / int(sample_rate / chunk * record_seconds)
        progress_bar.progress(progress)
    
    progress_bar.progress(1.0)
    st.success("錄音完成！")
    
    # 停止並關閉錄音串流
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # 儲存錄音檔案
    wf = wave.open(output_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    st.info(f"錄音已儲存至: {output_path}")
    return output_path

# 自定義鏈來處理RetrievalQA和LLMChain之間的輸入輸出差異
class CustomSequentialChain(Chain):
    analysis_chain: RetrievalQA
    feedback_chain: LLMChain
    verbose: bool = False
    
    @property
    def input_keys(self) -> List[str]:
        return ["input"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["output"]
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        current_input = inputs["input"]
        
        # 第一個鏈 (RetrievalQA) 的輸入和輸出處理
        if self.verbose:
            st.write("\n\n**> 正在執行第一個鏈...**")
            st.write(f"\n**> 輸入:**\n{current_input}")
        
        # RetrievalQA 需要 "query" 作為輸入鍵
        result = self.analysis_chain.invoke({"query": current_input})
        current_output = result["result"]
        
        if self.verbose:
            st.write(f"\n**> 輸出:**\n{current_output}")
        
        # 第二個鏈 (LLMChain) 的輸入和輸出處理
        if self.verbose:
            st.write("**> 正在执行反馈链...**")
        
        result = self.feedback_chain.invoke({"analysis_result": current_output})
        final_output = result["text"]
        
        if self.verbose:
            st.write(f"**> 最终反馈:**\n{final_output}")
        
        return {"output": final_output}

def process_audio(audio_path):
    with st.spinner("處理音訊中..."):
        # Transcribe the audio file
        transcript_path = lyrics_recognition.transcribe_audio(
            audio_path,
            output_dir="./analysis/whisper",
            model_size="medium",
            verbose=True
        )

        # Detect pitch
        pitch_result = pitch_detection.process_audio_file(audio_path, show_plot=False)
        pitch_path = pitch_result['output_path']

        # Process lyrics and note
        measurement_data = process_lyrics_note.find_most_frequent_note(pitch_path, transcript_path)

        # Save the result to a file
        os.makedirs('./cache', exist_ok=True)
        with open('./cache/note_analysis.json', 'w', encoding='utf-8') as f:
            f.write(measurement_data)
            
        return measurement_data

def note_to_midi_number(note_name):
    """Convert a note name (e.g., 'C4', 'D#3') to MIDI note number."""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Parse the note name and octave
    if len(note_name) == 2:
        note, octave = note_name[0], int(note_name[1])
    else:
        note, octave = note_name[0:2], int(note_name[2])
    
    # Calculate MIDI note number
    note_index = notes.index(note)
    midi_number = (octave + 1) * 12 + note_index
    
    return midi_number

def create_midi_from_notes_with_timestamps(notes, timestamps, output_file="output.mid", tempo=120):
    """
    Create a MIDI file from a list of note names with specific timestamps.
    
    Args:
        notes: List of note names (e.g., ['C4', 'D#4', 'F#4'])
        timestamps: List of tuples (start_time, end_time) in seconds
        output_file: Output MIDI file path
        tempo: Tempo in BPM (needed for MIDI file but won't affect actual timing)
    """
    # Create a MIDI file with one track
    midi = MIDIFile(1)
    track = 0
    
    # Set track name and tempo
    midi.addTrackName(track, 0, "Piano Track")
    midi.addTempo(track, 0, tempo)
    
    # Add notes to the track with specific timings
    for i, (note_name, (start_time, end_time)) in enumerate(zip(notes, timestamps)):
        try:
            midi_number = note_to_midi_number(note_name)
            # Convert seconds to beats (quarter notes)
            start_beats = start_time * (tempo / 60)
            duration_beats = (end_time - start_time) * (tempo / 60)
            midi.addNote(track, 0, midi_number, start_beats, duration_beats, 100)
        except ValueError:
            print(f"Skipping invalid note: {note_name}")
    
    # Write the MIDI file
    with open(output_file, "wb") as output_file:
        midi.writeFile(output_file)
    
    return output_file.name

def parse_timestamp(timestamp_str):
    """Convert a timestamp string like '00:00:01.920' to seconds."""
    h, m, s = timestamp_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def extract_timestamp_pairs(json_file_path):
    """Extract timestamp pairs from transcript JSON file."""
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Extract timestamp pairs
    timestamp_pairs = []
    for segment in data:
        start_time = segment["start_time"]
        end_time = segment["end_time"]
        start_seconds = parse_timestamp(start_time)
        end_seconds = parse_timestamp(end_time)
        timestamp_pairs.append((start_seconds, end_seconds))
    
    return timestamp_pairs

def notes_to_piano_with_timestamps(notes_string, timestamp_pairs, output_file="piano_output.mid", tempo=120):
    """
    Convert a string of comma-separated notes to a MIDI piano file with specific timestamps.
    
    Args:
        notes_string: Comma-separated string of notes (e.g., "C4, D#4, F#4")
        timestamp_pairs: List of timestamp pairs [(start_time, end_time), ...]
        output_file: Output MIDI file path
        tempo: Tempo in BPM (needed for MIDI file format but won't affect actual timing)
    """
    # Parse the notes string
    notes = [note.strip() for note in notes_string.split(',')]
    
    # Create the MIDI file
    midi_file = create_midi_from_notes_with_timestamps(notes, timestamp_pairs, output_file, tempo)
    
    return midi_file

def analyze_singing(measurement_data):
    with st.spinner("分析歌唱表現中..."):
        # Load the documents
        json_files = glob('./docs/*.json')
        raw_documents = []
        for file_path in json_files:
            loader = TextLoader(file_path)
            raw_documents.extend(loader.load())
        st.info(f"載入了 {len(raw_documents)} 個文件")

        # Split the documents
        text_splitter = CharacterTextSplitter(
            separator='\n\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        documents = text_splitter.split_documents(raw_documents)

        # Embedding
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )
        
        # Create a Chroma database
        db = Chroma.from_documents(raw_documents, embedding=OpenAIEmbeddings())

        # 創建分析音準鏈，加入檢索器
        retriever = db.as_retriever(search_kwargs={"k": 1})

        # 創建分析音準鏈的提示模板
        analysis_prompt_tpl = PromptTemplate.from_template(
            '去對比db中哪一段歌詞相似並列出來\n{context}\n'
            '接下來整理出唱歌數據與db相似歌詞所對應的音符的比較表格列出來。'
            '唱歌數據: {question}'
        )

        analysis_llm = ChatOpenAI(
            model_name='gpt-4o',
            temperature=0,
            max_tokens=1024
        )

        # 使用RetrievalQA替代LLMChain，並加入自定義提示模板
        analysis_chain = RetrievalQA.from_chain_type(
            llm=analysis_llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": analysis_prompt_tpl},
            return_source_documents=False
        )

        # 創建建議糾正鏈
        feedback_prompt_tpl = PromptTemplate.from_template(
            '你是一位專業的歌唱老師，根據你對歌唱的了解與音準分析，'
            '你的任務是看到哪一個音沒唱對請對照那個詞來說明音準要怎麼修正。'
            '生成對唱歌學生的建議，主要以鼓勵簡潔的方式告訴學生哪部分唱錯了需要調整。'
            '音準分析報告: {analysis_result}'
        )

        feedback_llm = ChatOpenAI(
            model_name='gpt-4o',
            temperature=0.6,
            max_tokens=1024
        )

        feedback_chain = LLMChain(llm=feedback_llm, prompt=feedback_prompt_tpl)

        # 使用新的自定義鏈
        chain = CustomSequentialChain(
            analysis_chain=analysis_chain,
            feedback_chain=feedback_chain,
            verbose=True
        )

        # 只使用學生的演唱數據作為輸入
        singing_data_str = json.dumps(measurement_data, ensure_ascii=False)
        
        # 獲取分析結果
        two_midi = analysis_chain.invoke({"query": singing_data_str})
        analysis_result = two_midi["result"]
        
        # 獲取參考音符序列
        ref_midi_tpl = PromptTemplate.from_template(
            '你是一個音樂分析助手。請分析以下唱歌數據，並從資料庫中找出相似的音符序列。'
            '唱歌數據: {query}'
            '請只輸出資料庫中相對應的音符序列，格式為逗號分隔的音符列表，不要包含歌詞或其他任何解釋。'
            '例如:C4, D4, E4, F4, G4'
            '你的回答應該只包含這個音符序列，不要有任何其他文字。'
        )
        ref_midi_llm = ChatOpenAI(
            model_name='gpt-4o',
            temperature=0,
            max_tokens=1024
        )
        ref_midi_chain = LLMChain(llm=ref_midi_llm, prompt=ref_midi_tpl)
        ref_midi = ref_midi_chain.invoke({"query": analysis_result})["text"]
        
        # 獲取學生唱的音符序列
        sing_midi_tpl = PromptTemplate.from_template(
            '你是一個音樂分析助手。請分析以下唱歌數據'
            '唱歌數據: {query}'
            '請只輸出音符，格式為逗號分隔的音符列表，不要包含歌詞或其他任何解釋。'
            '例如:C4, D4, E4, F4, G4'
            '你的回答應該只包含這個音符序列，不要有任何其他文字。'
        )
        sing_midi_llm = ChatOpenAI(
            model_name='gpt-4o',
            temperature=0,
            max_tokens=1024
        )
        sing_midi_chain = LLMChain(llm=sing_midi_llm, prompt=sing_midi_tpl)
        sing_midi = sing_midi_chain.invoke({"query": singing_data_str})["text"]
        
        # 使用自定義鏈處理輸入獲取反饋
        response = chain.invoke({"input": singing_data_str})
        feedback = response["output"]
        
        # 返回所有結果
        return {
            "feedback": feedback,
            "ref_midi": ref_midi,
            "sing_midi": sing_midi,
            "analysis_result": analysis_result
        }

def text_to_speech(text):
    # 將文字轉換為語音
    speech = gTTS(text=text, lang='zh-tw')
    
    # 產生唯一的檔名 (使用時間戳記)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"response/output_{timestamp}.mp3"
    
    # 確保輸出目錄存在
    os.makedirs("response", exist_ok=True)
    
    speech.save(output_path)
    return output_path


# Main app interface
st.sidebar.header("音訊輸入選項")
input_option = st.sidebar.radio("選擇音訊輸入方式:", ["上傳音檔", "即時錄音"])

# Audio input section
st.header("1. 音訊輸入")
col1, col2 = st.columns(2)

with col1:
    if input_option == "上傳音檔":
        uploaded_file = st.file_uploader("上傳音檔", type=['wav', 'mp3', 'm4a'])
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.audio_path = tmp_file.name
            st.success(f"音檔已上傳: {uploaded_file.name}")
            st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
    else:
        record_seconds = st.slider("錄音時間 (秒)", 5, 60, 30)
        if st.button("開始錄音"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"./audio/recorded_{timestamp}.wav"
            st.session_state.audio_path = record_audio(output_path, record_seconds)
            st.audio(st.session_state.audio_path)

with col2:
    if st.session_state.audio_path:
        st.info("音訊檔案已準備好")
        if st.button("分析歌唱表現"):
            measurement_data = process_audio(st.session_state.audio_path)
            
            # 修正轉錄文件路徑的獲取方式
            audio_filename = os.path.basename(st.session_state.audio_path).split('.')[0]
            possible_transcript_paths = [
                os.path.join("./analysis/whisper", f"{audio_filename}.json"),
                os.path.join("./analysis/whisper", f"{os.path.basename(st.session_state.audio_path)}.json")
            ]
            
            # 嘗試找到存在的轉錄文件
            transcript_path = None
            for path in possible_transcript_paths:
                if os.path.exists(path):
                    transcript_path = path
                    break
            
            # 如果找不到，嘗試查找whisper目錄中最新的json文件
            if not transcript_path:
                whisper_dir = "./analysis/whisper"
                if os.path.exists(whisper_dir):
                    json_files = [f for f in os.listdir(whisper_dir) if f.endswith('.json')]
                    if json_files:
                        # 按修改時間排序，取最新的
                        latest_file = max(json_files, key=lambda f: os.path.getmtime(os.path.join(whisper_dir, f)))
                        transcript_path = os.path.join(whisper_dir, latest_file)
            
            st.session_state.transcript_path = transcript_path
            
            if not transcript_path or not os.path.exists(transcript_path):
                st.error(f"找不到轉錄文件，請確保音頻處理正確完成")
                st.info("繼續進行分析，但無法生成MIDI對比")
            
            # 分析歌唱表現
            results = analyze_singing(measurement_data)
            st.session_state.feedback = results["feedback"]
            st.session_state.analysis_result = results["analysis_result"]
            st.session_state.audio_response = text_to_speech(st.session_state.feedback)
            
            # 生成MIDI文件並轉換為可播放的音頻
            if st.session_state.transcript_path and os.path.exists(st.session_state.transcript_path):
                try:
                    # 提取時間戳
                    timestamp_pairs = extract_timestamp_pairs(st.session_state.transcript_path)
                    
                    # 生成參考MIDI和學生MIDI
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # 確保目錄存在
                    os.makedirs("./midi", exist_ok=True)
                    os.makedirs("./audio/midi_audio", exist_ok=True)
                    
                    # 使用絕對路徑
                    ref_midi_path = os.path.abspath(f"./midi/ref_midi_{timestamp}.mid")
                    sing_midi_path = os.path.abspath(f"./midi/sing_midi_{timestamp}.mid")
                    
                    # 創建MIDI文件
                    st.session_state.ref_midi_path = notes_to_piano_with_timestamps(
                        results["ref_midi"], 
                        timestamp_pairs, 
                        ref_midi_path
                    )
                    
                    st.session_state.sing_midi_path = notes_to_piano_with_timestamps(
                        results["sing_midi"], 
                        timestamp_pairs, 
                        sing_midi_path
                    )
                    
                    # 檢查文件是否成功創建
                    if os.path.exists(st.session_state.ref_midi_path) and os.path.exists(st.session_state.sing_midi_path):
                        st.success("MIDI文件生成成功！")
                        
                        # 直接生成簡單的音頻文件
                        try:
                            # 確保輸出目錄存在
                            os.makedirs("./audio/simple_tones", exist_ok=True)
                            
                            # 定義輸出路徑
                            ref_audio_path = os.path.abspath(f"./audio/simple_tones/ref_tones_{timestamp}.wav")
                            sing_audio_path = os.path.abspath(f"./audio/simple_tones/sing_tones_{timestamp}.wav")
                            
                            # 使用簡單的方法生成音頻
                            def generate_simple_audio(notes_str, output_path, sample_rate=44100):
                                import numpy as np
                                from scipy.io import wavfile
                                
                                notes = [note.strip() for note in notes_str.split(',')]
                                duration = 0.5  # 每個音符的持續時間（秒）
                                
                                # 音符到頻率的映射
                                note_to_freq = {
                                    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63,
                                    'F4': 349.23, 'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00,
                                    'A#4': 466.16, 'B4': 493.88,
                                    'C5': 523.25, 'C#5': 554.37, 'D5': 587.33, 'D#5': 622.25, 'E5': 659.25,
                                    'F5': 698.46, 'F#5': 739.99, 'G5': 783.99, 'G#5': 830.61, 'A5': 880.00,
                                    'A#5': 932.33, 'B5': 987.77,
                                    'C3': 130.81, 'C#3': 138.59, 'D3': 146.83, 'D#3': 155.56, 'E3': 164.81,
                                    'F3': 174.61, 'F#3': 185.00, 'G3': 196.00, 'G#3': 207.65, 'A3': 220.00,
                                    'A#3': 233.08, 'B3': 246.94
                                }
                                
                                # 生成音頻數據
                                audio_data = np.array([], dtype=np.float32)
                                
                                for note in notes:
                                    if note in note_to_freq:
                                        freq = note_to_freq[note]
                                        t = np.linspace(0, duration, int(sample_rate * duration), False)
                                        tone = 0.5 * np.sin(2 * np.pi * freq * t)  # 生成正弦波
                                        audio_data = np.append(audio_data, tone)
                                    else:
                                        # 如果音符不在映射中，添加靜音
                                        silence = np.zeros(int(sample_rate * duration))
                                        audio_data = np.append(audio_data, silence)
                                
                                # 將浮點數據轉換為16位整數
                                audio_data = (audio_data * 32767).astype(np.int16)
                                
                                # 保存為WAV文件
                                wavfile.write(output_path, sample_rate, audio_data)
                                return output_path
                            
                            # 生成簡單的音頻文件
                            ref_audio_path = generate_simple_audio(results["ref_midi"], ref_audio_path)
                            sing_audio_path = generate_simple_audio(results["sing_midi"], sing_audio_path)
                            
                            # 保存音頻路徑
                            st.session_state.ref_audio_path = ref_audio_path
                            st.session_state.sing_audio_path = sing_audio_path
                            
                            st.success("已生成可播放的音頻文件！")
                        except Exception as e:
                            st.error(f"生成音頻時發生錯誤: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
                            
                            # 保留原始MIDI路徑
                            st.session_state.ref_audio_path = st.session_state.ref_midi_path
                            st.session_state.sing_audio_path = st.session_state.sing_midi_path
                    else:
                        st.error("MIDI文件創建失敗，請檢查文件路徑和權限")
                except Exception as e:
                    st.error(f"生成MIDI時發生錯誤: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

# Results section
if st.session_state.feedback:
    st.header("2. 分析結果")
    st.markdown(st.session_state.feedback)
    
    st.header("3. 語音回饋")
    st.audio(st.session_state.audio_response)
    
    # 使用轉換後的音頻文件路徑
    if hasattr(st.session_state, 'ref_audio_path') and hasattr(st.session_state, 'sing_audio_path'):
        st.header("4. 音符對比")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("標準音符")
            
            # 嘗試播放音頻
            if st.session_state.ref_audio_path.endswith(('.wav', '.mp3')):
                st.audio(st.session_state.ref_audio_path)
            else:
                st.warning("標準音符無法直接播放，請下載後使用音樂播放器播放")
                
            # 提供下載按鈕
            with open(st.session_state.ref_midi_path, "rb") as file:
                btn = st.download_button(
                    label="下載標準音符MIDI文件",
                    data=file,
                    file_name=os.path.basename(st.session_state.ref_midi_path),
                    mime="audio/midi"
                )
            
            # 顯示音符序列
            st.text_area("標準音符序列", results["ref_midi"], height=100)
        
        with col2:
            st.subheader("您的音符")
            
            # 嘗試播放音頻
            if st.session_state.sing_audio_path.endswith(('.wav', '.mp3')):
                st.audio(st.session_state.sing_audio_path)
            else:
                st.warning("您的音符無法直接播放，請下載後使用音樂播放器播放")
                
            # 提供下載按鈕
            with open(st.session_state.sing_midi_path, "rb") as file:
                btn = st.download_button(
                    label="下載您的音符MIDI文件",
                    data=file,
                    file_name=os.path.basename(st.session_state.sing_midi_path),
                    mime="audio/midi"
                )
            
            # 顯示音符序列
            st.text_area("您的音符序列", results["sing_midi"], height=100)
        
        # 顯示音符比較表格
        st.subheader("音符比較")
        ref_notes = [note.strip() for note in results["ref_midi"].split(',')]
        sing_notes = [note.strip() for note in results["sing_midi"].split(',')]
        
        # 確保兩個列表長度相同
        min_len = min(len(ref_notes), len(sing_notes))
        comparison_data = []
        
        for i in range(min_len):
            match = "✓" if ref_notes[i] == sing_notes[i] else "✗"
            comparison_data.append({
                "序號": i+1,
                "標準音符": ref_notes[i],
                "您的音符": sing_notes[i],
                "是否匹配": match
            })
        
        st.table(comparison_data)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("©2025 拯救胖虎大作戰")