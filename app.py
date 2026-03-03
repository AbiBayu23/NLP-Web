import streamlit as st
import re
import math
import PyPDF2
from docx import Document
import spacy
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import wordnet
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import pandas as pd
from collections import Counter
import altair as alt
import io
from fpdf import FPDF
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai



# ==========================================
# 1. KONFIGURASI HALAMAN WEB & TEMA
# ==========================================
st.set_page_config(
    page_title="AI NLP DOCUMENT ANALYSIS", 
    page_icon="☁️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp { background-color: #F8FAFC; }
        
        [data-testid="stFileUploadDropzone"] {
            background-color: #FFFFFF !important;
            border: 2px dashed #BAE6FD !important;
            border-radius: 10px !important;
        }
        [data-testid="stFileUploadDropzone"]:hover {
            border: 2px dashed #0EA5E9 !important;
            background-color: #F0F9FF !important;
        }        
        div[data-testid="stTextInput"] input { 
            color: #0F172A !important; 
            font-weight:500 !important; 
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #0EA5E9 !important;
            box-shadow: 0 0 0 1px #0EA5E9 !important;
        }        
        div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            cursor: pointer !important;
        }            
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stSelectbox"] {
            max-width: 140px !important;
            margin-left: auto !important;
        }        
        div[data-baseweb="select"] input {
            caret-color: transparent !important;
            cursor: pointer !important;
        }        
        button[data-baseweb="tab"] {
            background-color: transparent !important;
        }
    </style>
    <div style='background-color:#E0F2FE; padding:20px; border-radius:12px; border-left: 8px solid #0EA5E9; margin-bottom:25px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
        <h2 style='color:#0369A1; margin:0; font-weight: 800;'>☁️ Explorer NLP Documents Analysis</h2>
        <p style='color:#475569; margin:8px 0 0 0; font-size: 15px;'>Multi-File Management • Advanced Visualization • POS Search • Fast Extractive Summarization</p>
    </div>
""", unsafe_allow_html=True)

def render_dependency_tree(text):
    """Fungsi untuk merender pohon dependensi syntax beserta legenda keterangannya."""
    doc = nlp(text)
    
    options = {
        "compact": False,
        "distance": 100, 
        "arrow_stroke": 2, 
        "arrow_width": 6,
        "color": "#0284C7", 
        "bg": "#FFFFFF",
        "font": "Source Sans Pro"
    }
    svg_html = spacy.displacy.render(doc, style="dep", options=options, page=False)
    
    legenda = {
        "nsubj": "<b>Nominal Subject</b>: Pelaku/Subjek utama dalam kalimat.",
        "ROOT": "<b>Root</b>: Inti atau kata kerja utama dalam kalimat.",
        "obj / dobj": "<b>Object</b>: Penerima tindakan dari kata kerja.",
        "amod": "<b>Adjectival Modifier</b>: Kata sifat yang menjelaskan kata benda.",
        "advmod": "<b>Adverbial Modifier</b>: Keterangan tambahan (waktu/cara/tempat).",
        "det": "<b>Determiner</b>: Kata penentu seperti 'a', 'the', atau 'an'.",
        "prep": "<b>Preposition</b>: Kata depan yang menghubungkan frasa."
    }
    
    legenda_html = "".join([f"<li>{v}</li>" for v in legenda.values()])
    
    full_html = f"""
    <div style='border: 1px solid #BAE6FD; border-radius: 8px; background: #F8FAFC; padding: 15px; margin-top: 10px;'>
        <div style='overflow-x: auto; margin-bottom: 15px; background: white; border-radius: 6px; padding: 10px;'>
            {svg_html}
        </div>
        <div style='background: #E0F2FE; padding: 12px; border-radius: 6px;'>
            <p style='margin: 0 0 8px 0; font-weight: bold; color: #0369A1; font-size: 14px;'>💡 Cara Membaca Struktur (Dependency Tags):</p>
            <ul style='margin: 0; padding-left: 20px; font-size: 13px; color: #334155; line-height: 1.5;'>
                {legenda_html}
            </ul>
        </div>
    </div>
    """
    return full_html

DAFTAR_BAHASA = {
    'Indonesian': 'id', 'English': 'en', 'Spanish': 'es', 
    'French': 'fr', 'German': 'de', 'Japanese': 'ja', 'Korean': 'ko'
}

MAP_SEMUA_POS = {
    "NOUN (Kata Benda)": "NOUN",
    "VERB (Kata Kerja)": "VERB",
    "ADJ (Kata Sifat)": "ADJ",
    "ADV (Kata Keterangan)": "ADV",
    "PROPN (Nama/Entitas)": "PROPN",
    "PRON (Kata Ganti)": "PRON",
    "ADP (Preposisi/Kata Depan)": "ADP",
    "DET (Penentu/Determiner)": "DET",
    "AUX (Kata Bantu)": "AUX",
    "NUM (Angka)": "NUM",
    "PART (Partikel)": "PART",
    "SCONJ (Konjungsi Subordinatif)": "SCONJ",
    "CCONJ (Konjungsi Koordinatif)": "CCONJ",
    "INTJ (Interjeksi/Seruan)": "INTJ"
}

deskripsi_pos = {
    "NOUN": "Merujuk pada benda, manusia, tempat, atau ide abstrak (Contoh: buku, keadilan).",
    "VERB": "Menyatakan tindakan, proses, atau keadaan (Contoh: makan, berjalan, ada).",
    "ADJ": "Menjelaskan ciri, sifat, atau keadaan dari kata benda (Contoh: besar, pintar).",
    "ADV": "Memberikan keterangan tambahan pada verba atau adjektiva (Contoh: sangat, kemarin).",
    "PROPN": "Nama diri yang spesifik seperti nama orang, tempat, atau merek (Contoh: Jakarta, Budi).",
    "PRON": "Kata yang menggantikan penyebutan benda atau orang (Contoh: saya, mereka, itu).",
    "ADP": "Kata depan yang menunjukkan hubungan ruang atau waktu (Contoh: di, ke, dari).",
    "DET": "Kata yang memperjelas atau membatasi kata benda (Contoh: sebuah, setiap, ini).",
    "AUX": "Kata bantu yang mendampingi kata kerja utama (Contoh: telah, sedang, akan).",
    "NUM": "Menunjukkan jumlah, kuantitas, atau urutan angka (Contoh: satu, 2026, pertama).",
    "PART": "Kata tugas yang memiliki fungsi gramatikal khusus (Contoh: -lah, -kah, bukan).",
    "SCONJ": "Penghubung antara anak kalimat dan induk kalimat (Contoh: karena, jika, bahwa).",
    "CCONJ": "Penghubung dua unsur kalimat yang setara atau sejajar (Contoh: dan, atau, tetapi).",
    "INTJ": "Kata seru untuk mengungkapkan emosi, perasaan, atau sapaan (Contoh: wah, aduh, halo)."
}

Warna_POS_Utama = {
    'NOUN': '#2563EB', 'VERB': '#D97706', 'ADJ': '#059669', 'ADV': '#DC2626', 
    'PRON': '#7C3AED', 'PROPN': '#E11D48', 'ADP': '#475569', 'DET': '#0891B2', 
    'AUX': '#4F46E5', 'NUM': '#DB2777', 'PART': '#10B981', 'SCONJ': '#9333EA', 
    'CCONJ': '#C026D3', 'INTJ': '#EA580C'
}

@st.cache_resource
def siapkan_kamus_sinonim():
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        wordnet.ensure_loaded()
    except Exception as e:
        pass

siapkan_kamus_sinonim()

# ==========================================
# 2. CACHING AI & FUNGSI HELPER
# ==========================================
try:
    @spacy.Language.component("merge_hyphens")
    def merge_hyphens(doc):
        with doc.retokenize() as retokenizer:
            for match in re.finditer(r'\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)+\b', doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span is not None and len(span) > 1:
                    retokenizer.merge(span)
        return doc
except ValueError:
    pass

@st.cache_resource
def load_ai_models():
    try:
        nlp_model = spacy.load("en_core_web_sm")
    except:
        from spacy.lang.en import English
        nlp_model = English()
        nlp_model.add_pipe("sentencizer")
        
    if "merge_hyphens" not in nlp_model.pipe_names:
        if "tagger" in nlp_model.pipe_names:
            nlp_model.add_pipe("merge_hyphens", before="tagger")
        else:
            nlp_model.add_pipe("merge_hyphens")
            
    return nlp_model

with st.spinner("⏳ Membangunkan Engine NLP... Mohon tunggu sebentar."):
    nlp = load_ai_models()

@st.cache_data(show_spinner=False)
def dapatkan_data_visual(teks_terbatas):
    doc_vis = nlp(teks_terbatas)
    
    pos_counts = Counter([token.pos_ for token in doc_vis if token.pos_ not in ['SPACE', 'PUNCT', 'SYM', 'X']])
    df_pos = pd.DataFrame(pos_counts.most_common(), columns=['POS Tag', 'Jumlah Kata']) if pos_counts else pd.DataFrame()
    
    raw_words = re.findall(r'\b[a-z]+(?:-[a-z]+)*\b', teks_terbatas.lower())
    stop_words_spacy = nlp.Defaults.stop_words
    daftar_hitam_kustom = {
        'https', 'http', 'doi', 'et', 'al', 'www', 'com', 'org', 
        'pdf', 'fig', 'figure', 'table', 'vol', 'pp', 'ieee', 
        'the', 'and', 'for', 'that', 'using', 'based'
    }
    
    words_all = [w for w in raw_words if len(w) > 2 and w not in stop_words_spacy and w not in daftar_hitam_kustom]

    word_counts = Counter(words_all).most_common()
    df_words = pd.DataFrame(word_counts, columns=['Kata', 'Frekuensi']) if word_counts else pd.DataFrame()
    df_cloud = " ".join(words_all)
    
    return df_pos, df_words, df_cloud

def buat_word_sketch(kata_target, teks_mentah):
    """
    Mengekstrak Word Sketch (Profil Gramatikal) dari sebuah kata target.
    Mencari tahu kata ini sering dimodifikasi oleh apa, dan peran subjek/objeknya.
    """
    doc = nlp(teks_mentah[:100000]) 
    target_lemma = kata_target.lower()
    
    modifiers = []
    verbs_subject = []
    verbs_object = []
    
    for token in doc:
        if token.lemma_.lower() == target_lemma and not token.is_stop:
            for child in token.children:
                if child.dep_ in ['amod', 'advmod', 'compound'] and len(child.text) > 2:
                    modifiers.append(child.lemma_.lower())
            
            if token.dep_ in ['nsubj', 'nsubjpass']:
                if token.head.pos_ == 'VERB':
                    verbs_subject.append(token.head.lemma_.lower())
                    
            if token.dep_ in ['dobj', 'pobj']:
                head_token = token.head if token.dep_ == 'dobj' else token.head.head
                if head_token.pos_ == 'VERB':
                    verbs_object.append(head_token.lemma_.lower())
                    
    return {
        "✨ Modifiers (Sifat/Penjelas)": Counter(modifiers).most_common(5),
        "🏃‍♂️ Sebagai Subjek (Melakukan)": Counter(verbs_subject).most_common(5),
        "🎯 Sebagai Objek (Dikenai)": Counter(verbs_object).most_common(5)
    }

def hitung_collocation(kata_target, teks_mentah, window=3):
    words = re.findall(r'\b[a-z]+(?:-[a-z]+)*\b', teks_mentah.lower())
    stop_words = nlp.Defaults.stop_words
    pasangan = []

    for i, w in enumerate(words):
        if w == kata_target.lower():
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            for j in range(start, end):
                if i != j: 
                    kandidat = words[j]
                    if len(kandidat) > 2 and kandidat not in stop_words:
                        pasangan.append(kandidat)
                        
    return Counter(pasangan).most_common(5)



def get_colored_pos_text(text):
    doc = nlp(text)
    html = "<div style='line-height: 2.5; padding: 15px; background-color: #FFFFFF; border-radius: 8px; border: 1px solid #E2E8F0; box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.03);'>"
    for token in doc:
        bg_color = Warna_POS_Utama.get(token.pos_, '#94A3B8')
        html += f"<span style='background-color: {bg_color}; color: white; padding: 4px 10px; border-radius: 6px; margin: 3px; font-size: 14.5px; display: inline-block; font-weight:600; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>"
        html += f"{token.text} <span style='font-size: 10.5px; opacity: 0.9; margin-left: 4px; text-transform: uppercase;'>{token.pos_}</span></span> "
    html += "</div>"
    return html

Warna_NER = {
    'PERSON': '#F43F5E', 'ORG': '#8B5CF6', 'GPE': '#10B981', 'LOC': '#059669', 
    'DATE': '#F59E0B', 'TIME': '#FBBC05', 'MONEY': '#22C55E', 'PRODUCT': '#3B82F6'
}

def get_colored_ner_inline(text, target_label=None):
    doc = nlp(text)
    if not doc.ents: return text
    html = ""
    last_idx = 0
    for ent in doc.ents:
        html += text[last_idx:ent.start_char]
        bg_color = Warna_NER.get(ent.label_, "#3A4B62")
        opacity = "1.0" if (target_label is None or ent.label_ == target_label) else "0.4"
        html += f"<mark style='background-color: {bg_color}; color: white; padding: 2px 6px; border-radius: 4px; font-weight:600; font-size: 14.5px; opacity: {opacity}; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>{ent.text} <span style='font-size:10px; text-transform:uppercase;'>{ent.label_}</span></mark>"
        last_idx = ent.end_char
    html += text[last_idx:]
    return html

def extract_text(uploaded_file):
    text = ""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages: text += page.extract_text() + "\n"
        elif file_extension == 'docx':
            doc = Document(uploaded_file)
            for para in doc.paragraphs: text += para.text + "\n"
        elif file_extension == 'txt':
            text = uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
    return text

def terjemahkan_teks_panjang(teks, target_lang_code):
    """Fungsi pembagi teks agar tidak error limit 5000 karakter Google Translate"""
    paragraf = teks.split('\n')
    hasil = []
    for p in paragraf:
        if p.strip():
            if len(p) > 4900:
                potongan = [p[i:i+4900] for i in range(0, len(p), 4900)]
                for pot in potongan:
                    hasil.append(GoogleTranslator(source='auto', target=target_lang_code).translate(pot))
            else:
                hasil.append(GoogleTranslator(source='auto', target=target_lang_code).translate(p))
        else:
            hasil.append("")
    return '\n'.join(hasil)

if 'local_files' not in st.session_state: st.session_state.local_files = {}
if 'summary_results' not in st.session_state: st.session_state.summary_results = {}

def bersihkan_teks_untuk_analisis(teks_dokumen):
    """
    Membersihkan teks dari bagian yang tidak diinginkan khusus untuk proses NLP.
    """
    pola_referensi = re.compile(r'\n\s*(DAFTAR PUSTAKA|REFERENCES|BIBLIOGRAPHY|REFERENSI).*', re.IGNORECASE | re.DOTALL)
    teks = re.sub(pola_referensi, '', teks_dokumen)

    pola_awal = re.compile(r'.*?(?=BAB\s?I|INTRODUCTION|PENDAHULUAN)', re.IGNORECASE | re.DOTALL)
    if re.search(r'BAB\s?I|INTRODUCTION|PENDAHULUAN', teks, re.IGNORECASE):
        teks = re.sub(pola_awal, '', teks, count=1)

    teks = re.sub(r'(?m)^(Figure|Gambar|Table|Tabel|DOI|ISSN|Source).*$', '', teks)
    teks = re.sub(r'\[\d+\]', '', teks) 
    teks = re.sub(r'\(\w+ et al\., \d{4}\)', '', teks)

    return teks.strip()

# ==========================================
# 3. UI UPLOAD & MANAJEMEN FILE
# ==========================================

st.markdown("<h3 style='color:#0F172A;'>📁 Analisis Dokumen Eksternal</h3>", unsafe_allow_html=True)

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_files = st.file_uploader(
    "Upload File (PDF, DOCX, TXT)", 
    accept_multiple_files=True, 
    type=['pdf', 'docx', 'txt'],
    key=f"uploader_{st.session_state.uploader_key}"
)

if uploaded_files:
    ada_file_baru = False
    for file in uploaded_files:
        if file.name not in st.session_state.local_files:
            with st.spinner(f"Menganalisis {file.name}..."):
                raw_text = extract_text(file)
                teks_bersih = bersihkan_teks_untuk_analisis(raw_text)
                pola_kata = r'\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b'
                semua_kata = re.findall(pola_kata, raw_text.lower())
                
                st.session_state.local_files[file.name] = {
                    'text': raw_text,
                    'cleaned': teks_bersih,
                    'vocab': set(semua_kata), 
                    'stats': {
                        'k': len(list(nlp(raw_text[:80000]).sents)), 
                        'w': len(semua_kata)
                    }
                }
                ada_file_baru = True

    if ada_file_baru:
        st.session_state.uploader_key += 1
        st.rerun()

# ------------------------------------------
# KONTROL TAMPILAN KORPUS UTAMA
# ------------------------------------------
selected_files = [] 
if 'sub_corpora' not in st.session_state: 
    st.session_state.sub_corpora = {"General": []}

if st.session_state.local_files:
    file_names = list(st.session_state.local_files.keys())
    
    with st.expander("📂 Manajemen Korpus & Sub-Corpora", expanded=False):
        tab_manage, tab_grouping = st.tabs(["📄 Daftar File", "🗂️ Kelola Sub-Corpora"])
        
        # --- TAB 1: DAFTAR FILE ---
        with tab_manage:
            corpus_data = []
            for f_name, f_data in st.session_state.local_files.items():
                grup_file = [g for g, files in st.session_state.sub_corpora.items() if f_name in files]
                
                corpus_data.append({
                    "Hapus": False,
                    "Nama File": f_name,
                    "Sub-Corpus": ", ".join(grup_file) if grup_file else "Unassigned",
                    "Total Kata": f_data['stats']['w'],
                    "Kekayaan Kata (%)": round((len(f_data['vocab']) / f_data['stats']['w']) * 100, 2) if f_data['stats']['w'] > 0 else 0,
                })
            
            df_corpus = pd.DataFrame(corpus_data)
            edited_df = st.data_editor(
                df_corpus,
                column_config={
                    "Hapus": st.column_config.CheckboxColumn("❌ Hapus", default=False),
                    "Nama File": st.column_config.TextColumn("📄 Dokumen", disabled=True, width="medium"),
                    "Sub-Corpus": st.column_config.TextColumn("🗂️ Group", disabled=True),
                    "Kekayaan Kata (%)": st.column_config.ProgressColumn("💎 Richness", format="%.2f%%", min_value=0, max_value=100),
                },
                hide_index=True, use_container_width=True
            )
            
            file_untuk_dihapus = edited_df[edited_df["Hapus"] == True]["Nama File"].tolist()
            if st.button("🗑️ Hapus File Terpilih", type="primary", disabled=not file_untuk_dihapus):
                for fname in file_untuk_dihapus:
                    del st.session_state.local_files[fname]
                    for g in st.session_state.sub_corpora:
                        if fname in st.session_state.sub_corpora[g]:
                            st.session_state.sub_corpora[g].remove(fname)
                st.rerun()

        # --- TAB 2: KELOLA SUB-CORPORA ---
        with tab_grouping:
            c1, c2 = st.columns([1, 1])
            with c1:
                new_group = st.text_input("Buat Sub-Corpus Baru:", placeholder="Contoh: Paper_NLP_2026")
                if st.button("➕ Tambah Grup") and new_group:
                    if new_group not in st.session_state.sub_corpora:
                        st.session_state.sub_corpora[new_group] = []
                        st.success(f"Grup '{new_group}' dibuat!")
            
            with c2:
                target_grup = st.selectbox("Pilih Grup Tujuan:", list(st.session_state.sub_corpora.keys()))
                files_to_add = st.multiselect("Pilih File untuk Dimasukkan:", file_names)
                if st.button("📥 Masukkan ke Grup"):
                    for f in files_to_add:
                        if f not in st.session_state.sub_corpora[target_grup]:
                            st.session_state.sub_corpora[target_grup].append(f)
                    st.toast(f"Berhasil update grup {target_grup}!")
                    st.rerun()

    # --- FILTER PEMILIHAN FILE BERDASARKAN GRUP ---
    st.markdown("<div style='font-size:14px; font-weight:600; color:#475569; margin-bottom:8px;'>Filter Dokumen Aktif:</div>", unsafe_allow_html=True)
    
    col_filter_1, col_filter_2 = st.columns([4, 6])
    with col_filter_1:
        opsi_grup = ["Semua File"] + list(st.session_state.sub_corpora.keys())
        pilihan_grup = st.selectbox("Berdasarkan Sub-Corpus:", opsi_grup, label_visibility="collapsed")
    
    with col_filter_2:
        if pilihan_grup == "Semua File":
            default_files = file_names
        else:
            default_files = st.session_state.sub_corpora[pilihan_grup]
            
        selected_files = st.multiselect(
            "File Terpilih:", 
            options=file_names, 
            default=default_files,
            label_visibility="collapsed"
        )

    col_sel1, col_sel2 = st.columns([8, 2], vertical_alignment="bottom")
    
    with col_sel1:
        st.markdown("<div style='font-size:14px; font-weight:600; color:#475569; margin-bottom:8px;'>Pilih Dokumen Aktif:</div>", unsafe_allow_html=True)
        is_all = st.session_state.get('pilih_semua_check', False)
        selected_files = st.multiselect(
            "Pilih Dokumen Aktif:", 
            options=file_names, 
            default=file_names if is_all else [file_names[-1]],
            key="ms_files",
            label_visibility="collapsed"
        )
    
    with col_sel2:
        st.checkbox("Pilih Semua", key="pilih_semua_check")

    if selected_files:
        
        # ==========================================
        # 5. TAB FITUR NLP (PUSAT ANALISIS TERPADU)
        # ==========================================
        @st.cache_data(show_spinner=False)
        def get_cached_wordcloud(text_data):
            wc = WordCloud(width=600, height=280, background_color='white', colormap='viridis', max_words=300).generate(text_data)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            return fig

        @st.fragment
        def fitur_nlp_dashboard(selected_files):

            def prev_page_1(): st.session_state.current_page -= 1
            def next_page_1(): st.session_state.current_page += 1
            def update_ws_word(new_word): st.session_state["in_word_sketch"] = new_word 

            tab_compare_vis, tab_search, tab_summary = st.tabs([
                "⚖️ Compare", 
                "🔍 Searching", 
                "📝 Summarize"
            ])

            # ==========================================
            # TAB 1: VISUALISASI & PERBANDINGAN DOKUMEN
            # ==========================================
            with tab_compare_vis:
                st.markdown("<h3 style='color:#0F172A;'>⚖️ Perbandingan & Visualisasi Dokumen</h3>", unsafe_allow_html=True)
                
                with st.expander("📖 Panduan: Cara Membaca Fitur Ini"):
                    st.info("""
                    **Bagian ini membandingkan dua dokumen secara langsung dan menampilkan visualisasi khusus untuk keduanya.**
                    
                    **Metrik Perbandingan:**
                    1. **📊 Jaccard Index:** Persentase kata yang **sama persis** di antara kedua dokumen. 
                    2. **🧠 Cosine Similarity (TF-IDF):** Kemiripan **Makna/Topik** secara matematis menggunakan AI Embedding.
                    
                    *Setelah memilih dokumen, gulir ke bawah untuk melihat grafik tata bahasa, Word Cloud, dan N-Grams masing-masing dokumen.*
                    """)
                
                # CSS untuk membuat Scrollbar Vertikal di area visualisasi
                st.markdown("""
                    <style>
                    .scrollable-column {
                        max-height: 800px;
                        overflow-y: auto;
                        padding-right: 15px;
                        scrollbar-width: thin;
                    }
                    .scrollable-column::-webkit-scrollbar {
                        width: 6px;
                    }
                    .scrollable-column::-webkit-scrollbar-track {
                        background: #F1F5F9; 
                        border-radius: 4px;
                    }
                    .scrollable-column::-webkit-scrollbar-thumb {
                        background: #CBD5E1; 
                        border-radius: 4px;
                    }
                    .scrollable-column::-webkit-scrollbar-thumb:hover {
                        background: #94A3B8; 
                    }
                    </style>
                """, unsafe_allow_html=True)

                opsi_semua_doc = list(st.session_state.local_files.keys())
                docs_to_visualize = []

                if len(opsi_semua_doc) == 0:
                    st.warning("Silakan unggah dokumen terlebih dahulu.")
                
                elif len(opsi_semua_doc) == 1:
                    # Jika cuma 1 file yang diupload, langsung tampilkan visualisasinya tanpa komparasi
                    st.info("ℹ️ Anda baru mengunggah 1 dokumen. Tambahkan minimal 2 dokumen untuk menggunakan fitur Komparasi. Menampilkan visualisasi tunggal:")
                    doc_a = opsi_semua_doc[0]
                    docs_to_visualize = [doc_a]
                
                else:
                    # Jika >= 2 file, jalankan form komparasi
                    with st.container(border=True):
                        col_comp1, col_comp2 = st.columns(2)
                        
                        with col_comp1: doc_a = st.selectbox("Pilih Dokumen A:", opsi_semua_doc, key="comp_doc_a")
                        with col_comp2: doc_b = st.selectbox("Pilih Dokumen B:", opsi_semua_doc, index=(1 if len(opsi_semua_doc) > 1 else 0), key="comp_doc_b")
                        
                        if doc_a and doc_b:
                            if doc_a == doc_b: 
                                st.warning("⚠️ Silakan pilih dua dokumen yang berbeda untuk dibandingkan.")
                                docs_to_visualize = [] # Jangan render visualisasi jika filenya sama
                            else:
                                docs_to_visualize = [doc_a, doc_b]
                                with st.spinner("Menghitung Vektor Kemiripan AI (Document Embedding)..."):
                                    from sklearn.feature_extraction.text import TfidfVectorizer
                                    from sklearn.metrics.pairwise import cosine_similarity
                                    
                                    vocab_a = st.session_state.local_files[doc_a]['vocab']
                                    vocab_b = st.session_state.local_files[doc_b]['vocab']
                                    
                                    irisan = vocab_a.intersection(vocab_b)
                                    gabungan = vocab_a.union(vocab_b)
                                    jaccard_sim = len(irisan) / len(gabungan) if len(gabungan) > 0 else 0
                                    
                                    teks_a = st.session_state.local_files[doc_a]['cleaned']
                                    teks_b = st.session_state.local_files[doc_b]['cleaned']
                                    
                                    vectorizer = TfidfVectorizer(stop_words='english')
                                    tfidf_matrix = vectorizer.fit_transform([teks_a, teks_b])
                                    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                                    
                                    unik_a = vocab_a - vocab_b
                                    unik_b = vocab_b - vocab_a
                                    
                                    raw_a = re.findall(r'\b[a-z]+(?:-[a-z]+)*\b', teks_a.lower())
                                    raw_b = re.findall(r'\b[a-z]+(?:-[a-z]+)*\b', teks_b.lower())
                                    
                                    stop_words = nlp.Defaults.stop_words
                                    count_unik_a = Counter([w for w in raw_a if w in unik_a and w not in stop_words and len(w) > 2])
                                    count_unik_b = Counter([w for w in raw_b if w in unik_b and w not in stop_words and len(w) > 2])
                                    
                                    st.markdown("---")
                                    col_score1, col_score2 = st.columns(2)
                                    
                                    with col_score1:
                                        st.markdown(f"""
                                        <div style='background:#F0F9FF; padding:20px; border-radius:12px; border:1px solid #BAE6FD; text-align:center; height:100%;'>
                                            <div style='font-size:15px; color:#0369A1; font-weight:700; margin-bottom:5px;'>📊 Jaccard Index</div>
                                            <div style='font-size:13px; color:#475569; margin-bottom:10px;'>(Kemiripan Kosakata Eksak)</div>
                                            <div style='font-size:42px; color:#0EA5E9; font-weight:900;'>{jaccard_sim*100:.1f}%</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                    with col_score2:
                                        st.markdown(f"""
                                        <div style='background:#FAF5FF; padding:20px; border-radius:12px; border:1px solid #E9D5FF; text-align:center; height:100%; box-shadow: 0 4px 6px -1px rgba(168, 85, 247, 0.1);'>
                                            <div style='font-size:15px; color:#6B21A8; font-weight:700; margin-bottom:5px;'>🧠 Cosine Similarity</div>
                                            <div style='font-size:13px; color:#475569; margin-bottom:10px;'>(Kemiripan Makna via TF-IDF Vektor)</div>
                                            <div style='font-size:42px; color:#A855F7; font-weight:900;'>{cosine_sim*100:.1f}%</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    st.write("")
                                    c1, c2, c3 = st.columns(3)
                                    c1.metric(f"Total Kata Unik ({doc_a})", f"{len(vocab_a):,}")
                                    c2.metric("Kata Beririsan (Sama)", f"{len(irisan):,}")
                                    c3.metric(f"Total Kata Unik ({doc_b})", f"{len(vocab_b):,}")
                                    
                                    st.write("")
                                    col_res1, col_res2 = st.columns(2)
                                    
                                    def render_unique_words(doc_name, count_data, color_bg, color_border, icon):
                                        top_unik = count_data.most_common(30)
                                        pills_html = f"<div style='display:flex; gap:8px; flex-wrap:wrap; margin-top:20px;'>{''.join([f'<span style=\"background:white; color:#0F172A; border:1px solid {color_border}; border-radius:15px; padding:4px 12px; font-size:12.5px; font-weight:500; box-shadow: 0 1px 2px rgba(0,0,0,0.05);\">{w} <span style=\"color:#94A3B8; font-size:10.5px; margin-left:3px;\">({freq}x)</span></span>' for w, freq in top_unik])}</div>" if top_unik else "<p style='font-style:italic; color:#64748B; margin-top:15px;'>Tidak ada kata eksklusif yang signifikan.</p>"
                                        st.markdown(f"<div style='background:{color_bg}; padding:25px; border-radius:12px; border:1px solid {color_border}; height: 100%;'><h4 style='margin:0 0 8px 0; color:#0F172A; font-size:16.5px; font-weight:700;'>{icon} Eksklusif di Dokumen Ini</h4><p style='margin:0; font-size:13.5px; color:#475569; line-height:1.5;'>Sering dibahas dalam <b>{doc_name}</b>, namun <u>tidak pernah</u> disebutkan di sebelahnya.</p>{pills_html}</div>", unsafe_allow_html=True)

                                    with col_res1: render_unique_words("Dokumen A", count_unik_a, "#F0F9FF", "#BAE6FD", "📘")
                                    with col_res2: render_unique_words("Dokumen B", count_unik_b, "#FDF4FF", "#FBCFE8", "📙")

                                    # ==========================================
                                    # KESIMPULAN UNIK DENGAN GENERATIVE AI (AUTO-DETECT MODEL)
                                    # ==========================================
                                    st.write("") # Spacer
                                    
                                    with st.spinner("✨ AI Generatif sedang merenungkan hasil dan menyusun narasi kesimpulan..."):
                                        try:
                                            import google.generativeai as genai
                                            genai.configure(api_key="AIzaSyCUeZt5KYx769PwTsIRnmfHT66Rxpuu994") 
                                            
                                            # 1. AUTO-DETECT MODEL TERBAIK (Anti-Error 404)
                                            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                                            
                                            # Prioritaskan model gemini 1.5, jika tidak ada cari gemini biasa, jika tidak ada ambil apapun yang tersedia
                                            target_model_name = next((m for m in available_models if 'gemini-1.5' in m), None) or \
                                                                next((m for m in available_models if 'gemini' in m), available_models[0])
                                            
                                            model_ai = genai.GenerativeModel(target_model_name)
                                            
                                            # 2. Ambil top 3 kata untuk dimasukkan ke prompt
                                            top3_a = [w for w, f in count_unik_a.most_common(3)]
                                            top3_b = [w for w, f in count_unik_b.most_common(3)]
                                            
                                            # 3. Susun Prompt (Perintah) ke AI
                                            prompt_instruksi = f"""
                                            Kamu adalah seorang analis linguistik profesional. Tugasmu adalah memberikan satu paragraf kesimpulan analitis yang mendalam, natural, dan tidak seperti robot.
                                            
                                            Berikut adalah data perbandingan dari dua dokumen yang baru saja diproses:
                                            1. Kemiripan Makna/Topik (Cosine Similarity): {cosine_sim*100:.1f}%
                                            2. Kemiripan Kosakata yang sama persis (Jaccard Index): {jaccard_sim*100:.1f}%
                                            3. Dokumen A sangat fokus pada kata kunci: {', '.join(top3_a) if top3_a else 'tidak ada dominasi spesifik'}
                                            4. Dokumen B sangat fokus pada kata kunci: {', '.join(top3_b) if top3_b else 'tidak ada dominasi spesifik'}
                                            
                                            Instruksi:
                                            - Berikan kesimpulan apa arti dari angka-angka di atas. Apakah mereka membahas topik yang sama tapi gaya bahasa beda? Atau topiknya memang beda?
                                            - Jangan gunakan format list/bullet point (1, 2, 3). Jadikan satu narasi paragraf yang mengalir estetik.
                                            - Hasil harus selalu BEDA dan UNIK setiap kali kamu merespons. Jangan sebutkan instruksi ini di jawabanmu.
                                            """
                                            
                                            # 4. Generate respons
                                            respons_genai = model_ai.generate_content(prompt_instruksi)
                                            narasi_kesimpulan = respons_genai.text
                                            
                                            # Bersihkan nama model untuk ditampilkan di UI
                                            nama_model_bersih = target_model_name.replace('models/', '')
                                            
                                            # 5. Render ke layar
                                            st.markdown(f"""
<div style='background-color: #ECFDF5; border-left: 6px solid #10B981; padding: 25px; border-radius: 8px; margin-top: 25px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);'>
    <h4 style='color: #065F46; margin-top: 0; margin-bottom: 15px;'>✨ Insight Analisis Generatif <span style='font-size:12px; color:#10B981;'>({nama_model_bersih})</span></h4>
    <p style='color: #064E3B; font-size:15px; line-height:1.8; margin-bottom:0; text-align: justify;'>
        {narasi_kesimpulan}
    </p>
</div>
                                            """, unsafe_allow_html=True)
                                            
                                        except Exception as e:
                                            st.error(f"Gagal menghasilkan kesimpulan AI. Error: {e}")
                                            
                # ====================================================
                # VISUALISASI RENDER FUNGSI (Untuk 1 atau 2 Dokumen)
                # ====================================================
                if docs_to_visualize:
                    st.markdown("<br><h4 style='color:#0F172A;'>📈 Visualisasi Detail Dokumen</h4>", unsafe_allow_html=True)
                    
                    # Fungsi Pembantu untuk merender visualisasi dalam format vertikal yang bisa di-scroll
                    def render_visualisasi_sisi(fname, warna_utama, warna_kedua):
                        st.markdown(f"""
                            <div style='background:{warna_utama}22; padding:10px 15px; border-radius:8px; border-bottom:3px solid {warna_utama}; margin-bottom:15px;'>
                                <h4 style='margin:0; color:#1E293B;'>📄 {fname}</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<div class='scrollable-column'>", unsafe_allow_html=True)
                        
                        if 'vis_cache' not in st.session_state.local_files[fname]:
                            with st.spinner(f"Memproses visualisasi {fname}..."):
                                teks_dokumen = st.session_state.local_files[fname]['text']
                                df_pos, df_words, df_cloud_text = dapatkan_data_visual(teks_dokumen[:80000])
                                teks_mentah_aktif = st.session_state.local_files[fname]['cleaned']
                                
                                if not df_words.empty and 'Pasangan 1' not in df_words.columns:
                                    col1, col2, col3, col4, col5 = [], [], [], [], []
                                    for i, kata in enumerate(df_words['Kata']):
                                        if i < 500:  
                                            hasil_colloc = hitung_collocation(kata, teks_mentah_aktif, window=5)
                                            for col, idx in zip([col1, col2, col3, col4, col5], range(5)):
                                                col.append(f"{hasil_colloc[idx][0]} ({hasil_colloc[idx][1]}x)" if len(hasil_colloc) > idx else "-")
                                        else:
                                            for col in [col1, col2, col3, col4, col5]: col.append("-")
                                    df_words['Pasangan 1'] = col1; df_words['Pasangan 2'] = col2; df_words['Pasangan 3'] = col3; df_words['Pasangan 4'] = col4; df_words['Pasangan 5'] = col5

                                words_ng = [w for w in re.findall(r'\b[a-z]{3,}\b', teks_mentah_aktif.lower()) if w not in nlp.Defaults.stop_words]
                                bigrams = [" ".join(g) for g in zip(*[words_ng[i:] for i in range(2)])]
                                trigrams = [" ".join(g) for g in zip(*[words_ng[i:] for i in range(3)])]
                                df_bigram = pd.DataFrame(Counter(bigrams).most_common(15), columns=['Frasa', 'Frekuensi'])
                                df_trigram = pd.DataFrame(Counter(trigrams).most_common(15), columns=['Frasa', 'Frekuensi'])

                                cloud_img = None
                                if df_cloud_text:
                                    fig = get_cached_wordcloud(df_cloud_text[:80000])
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
                                    plt.close(fig)
                                    cloud_img = buf.getvalue()

                                st.session_state.local_files[fname]['vis_cache'] = {
                                    'df_pos': df_pos, 'df_words': df_words, 'df_bigram': df_bigram, 'df_trigram': df_trigram, 'cloud_img': cloud_img
                                }

                        cache_vis = st.session_state.local_files[fname]['vis_cache']

                        st.markdown(f"**☁️ Word Cloud**")
                        if cache_vis['cloud_img']: st.image(cache_vis['cloud_img'], use_container_width=True)
                        st.write("---")

                        st.markdown(f"**📈 15 Kata Teratas**")
                        if not cache_vis['df_words'].empty: 
                            ch_kata = alt.Chart(cache_vis['df_words'].head(15)).mark_bar(color=warna_utama, cornerRadiusEnd=4).encode(
                                y=alt.Y('Kata:N', sort='-x', title=None), x=alt.X('Frekuensi:Q', title=None), 
                                tooltip=[alt.Tooltip('Kata:N'), alt.Tooltip('Frekuensi:Q'), alt.Tooltip('Pasangan 1:N', title='🔗 Colloc 1'), alt.Tooltip('Pasangan 2:N', title='🔗 Colloc 2')]
                            ).properties(height=350).configure_view(stroke='#94A3B8', strokeWidth=1)
                            st.altair_chart(ch_kata, use_container_width=True)
                        st.write("---")

                        st.markdown(f"**🔗 15 Frasa Bigram (2 Kata)**")
                        if not cache_vis['df_bigram'].empty: 
                            ch_bi = alt.Chart(cache_vis['df_bigram']).mark_bar(color=warna_kedua, cornerRadiusEnd=4).encode(
                                y=alt.Y('Frasa:N', sort='-x', title=None), x=alt.X('Frekuensi:Q', title=None), tooltip=['Frasa', 'Frekuensi']
                            ).properties(height=350).configure_view(stroke='#94A3B8', strokeWidth=1)
                            st.altair_chart(ch_bi, use_container_width=True)
                        st.write("---")

                        st.markdown(f"**🔗 15 Frasa Trigram (3 Kata)**")
                        if not cache_vis['df_trigram'].empty: 
                            ch_tri = alt.Chart(cache_vis['df_trigram']).mark_bar(color=warna_utama, opacity=0.8, cornerRadiusEnd=4).encode(
                                y=alt.Y('Frasa:N', sort='-x', title=None), x=alt.X('Frekuensi:Q', title=None), tooltip=['Frasa', 'Frekuensi']
                            ).properties(height=350).configure_view(stroke='#94A3B8', strokeWidth=1)
                            st.altair_chart(ch_tri, use_container_width=True)
                        st.write("---")

                        st.markdown(f"**🔤 Statistik Tata Bahasa (POS)**")
                        if not cache_vis['df_pos'].empty: 
                            chart_pos = alt.Chart(cache_vis['df_pos']).mark_bar(color=warna_kedua, opacity=0.8, cornerRadiusEnd=4).encode(
                                y=alt.Y('POS Tag', sort='-x', title=None), x=alt.X('Jumlah Kata', title=None), tooltip=['POS Tag', 'Jumlah Kata']
                            ).properties(height=350).configure_view(stroke='#94A3B8', strokeWidth=1)
                            st.altair_chart(chart_pos, use_container_width=True)
                        
                        if not cache_vis['df_words'].empty:
                            st.write("---")
                            st.markdown(f"**🗃️ Tabel Data Concordance (Top 500)**")
                            df_tabel_500 = cache_vis['df_words'].head(500)
                            st.dataframe(df_tabel_500, use_container_width=True, height=250, hide_index=True)
                            st.download_button(label="📥 Download Data CSV", data=df_tabel_500.to_csv(index=False).encode('utf-8'), file_name=f"concordance_{fname}.csv", mime="text/csv", key=f"dl_csv_{fname}")

                        st.markdown("</div>", unsafe_allow_html=True)

                    # LOGIKA RENDER TAMPILAN
                    if len(docs_to_visualize) == 1:
                        # Tampilan Lebar Penuh Jika Hanya 1 Dokumen Aktif
                        with st.container(border=True):
                            render_visualisasi_sisi(docs_to_visualize[0], warna_utama="#0EA5E9", warna_kedua="#38BDF8")
                    elif len(docs_to_visualize) == 2:
                        # Tampilan Side-by-Side Jika 2 Dokumen Aktif
                        with st.container(border=True):
                            col_vis_kiri, col_vis_kanan = st.columns(2)
                            with col_vis_kiri:
                                render_visualisasi_sisi(docs_to_visualize[0], warna_utama="#0EA5E9", warna_kedua="#38BDF8") # Tema Biru untuk Doc A
                            with col_vis_kanan:
                                render_visualisasi_sisi(docs_to_visualize[1], warna_utama="#D946EF", warna_kedua="#E879F9") # Tema Ungu untuk Doc B

            # ==========================================
            # TAB 2: PENCARIAN PINTAR & WORD SKETCH
            # ==========================================
            with tab_search:
                st.markdown("<h3 style='color:#0F172A;'>🔍 Pencarian Pintar & Pemrofilan Kata</h3>", unsafe_allow_html=True)
                
                with st.expander("📖 Panduan: Penjelasan Tiap Mode Pencarian"):
                    st.info("""
                        * **🔍 Lemmatization:** Mencari kata berdasarkan akar katanya. (Mencari *'analyze'* memunculkan *'analyzing'*, *'analyzes'*).
                        * **🔬 Morphology Search:** Menemukan variasi bentuk imbuhan dari satu kata dasar dan memetakannya ke grafik.
                        * **🧠 Semantic Search:** Mencari kalimat yang memiliki makna mirip dengan kata kunci.
                        * **🧩 Word Sketch (Profil Kata):** Mengetahui bagaimana sebuah kata dikelilingi oleh kata lain (Modifier, Subjek, Objek).
                        * **🏷️ Entity Search (NER):** Mencari kemunculan entitas seperti Nama Orang, Organisasi, atau Lokasi.
                        * **📚 POS Search:** Mencari kata berdasarkan jabatan tata bahasanya (Misal: *'record'* sebagai Kata Benda).
                        * **🛒 Boolean Search:** Pencarian menggunakan logika (*machine AND translation NOT neural*).
                        * **⚙️ Regex Search:** Pencarian pola kode (Sitasi, Email, dll).
                        * **🌳 Dependency Search:** Mencari sepasang kata yang berhubungan langsung di pohon sintaksis.
                        """)

                col_mode, col_input, col_btn = st.columns([2.5, 4.5, 1], gap="small")
                
                with col_mode:
                    mode_pencarian = st.selectbox("Mode Pencarian", [
                        "🔍 Lemmatization", "🔬 Morphology Search (Variasi Kata)", "🧠 Semantic Search", "🧩 Word Sketch (Profil Kata)", 
                        "🏷️ Entity Search (NER)", "📚 POS Search (Kelas Kata)", "🛒 Boolean Search", "⚙️ Regex Search", "🌳 Dependency Search"
                    ], label_visibility="collapsed")
                    
                with col_input:
                    if "NER" in mode_pencarian:
                        query_aktif = st.selectbox("Pilih Tipe Entitas", ["PERSON (Orang)", "ORG (Organisasi)", "GPE (Negara/Kota)", "LOC (Lokasi)", "DATE (Tanggal)", "MONEY (Keuangan)"], key="input_search_ner", label_visibility="collapsed")
                    elif "POS Search" in mode_pencarian:
                        c_pos1, c_pos2 = st.columns([1, 1])
                        with c_pos1:
                            pos_label = st.selectbox("Pilih Kelas Kata:", list(MAP_SEMUA_POS.keys()), key="pos_tag_sel", label_visibility="collapsed")
                            target_pos_tag = MAP_SEMUA_POS[pos_label]
                        with c_pos2: pos_keyword = st.text_input("Kata Kunci (Opsional):", placeholder="Ketik kata...", key="pos_kw_in", label_visibility="collapsed").strip().lower()
                        query_aktif = f"POS_{target_pos_tag}_{pos_keyword}"
                    elif "Dependency" in mode_pencarian:
                        c_dep1, c_dep2, c_dep3 = st.columns(3)
                        with c_dep1: head_word = st.text_input("Head (Induk):", placeholder="e.g. train", key="dep_h").strip().lower()
                        with c_dep2: rel_type = st.selectbox("Relation:", ["nsubj", "obj", "dobj", "amod", "advmod", "compound", "prep"], key="dep_r")
                        with c_dep3: child_word = st.text_input("Child (Anak):", placeholder="e.g. model", key="dep_c").strip().lower()
                        query_aktif = f"{head_word} {rel_type} {child_word}".strip()
                    elif "Regex" in mode_pencarian:
                        opsi_regex = {"📖 Sitasi (Penulis, Tahun)": r"\([A-Z][A-Za-z\s]+(?:et al\.)?,\s?\d{4}\)", "🔖 Referensi Gambar/Tabel": r"\b(?:Gambar|Figure|Tabel|Table)\s+\d+(?:\.\d+)*\b", "📧 Format Email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "✏️ Ketik Manual...": "manual"}
                        pilihan_regex = st.selectbox("Pilih Pola Regex", list(opsi_regex.keys()), label_visibility="collapsed")
                        if pilihan_regex == "✏️ Ketik Manual...": query_aktif = st.text_input("Ketik pola Regex", placeholder="Contoh: \\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b", label_visibility="collapsed")
                        else: query_aktif = opsi_regex[pilihan_regex]
                    elif "Word Sketch" in mode_pencarian:
                        query_aktif = st.text_input("Ketik kata untuk diprofilkan (huruf kecil):", placeholder="Contoh: model, data, machine...", key="in_word_sketch", label_visibility="collapsed").strip().lower()
                    elif "Semantic" in mode_pencarian: query_aktif = st.text_input("Ketik konsep makna", placeholder="Ketik kata (misal: technology)...", label_visibility="collapsed").strip()
                    elif "Lemmatization" in mode_pencarian: query_aktif = st.text_input("Ketik kata dasar", placeholder="Ketik kata dasar (misal: analyze)...", label_visibility="collapsed").strip()
                    elif "Morphology" in mode_pencarian: query_aktif = st.text_input("Ketik Akar Kata", placeholder="Ketik akar kata (misal: develop, use)...", label_visibility="collapsed").strip()
                    elif "Boolean" in mode_pencarian: query_aktif = st.text_input("Query Boolean", placeholder="Contoh: machine AND translation NOT neural", label_visibility="collapsed")

                with col_btn:
                    if "Regex" in mode_pencarian and query_aktif != "manual" and query_aktif != "": st.markdown("<div style='margin-top: 2px;'></div>", unsafe_allow_html=True)
                    btn_cari = st.button("Cari", key="btn_search_key", use_container_width=True, type="primary")

                if 'search_results' not in st.session_state: st.session_state.search_results = []
                if 'morph_results' not in st.session_state: st.session_state.morph_results = pd.DataFrame()
                if 'sketch_results' not in st.session_state: st.session_state.sketch_results = {}
                if 'current_page' not in st.session_state: st.session_state.current_page = 0
                if 'last_query' not in st.session_state: st.session_state.last_query = ""

                # --- PROSES PENCARIAN UTAMA ---
                if query_aktif:
                    if st.session_state.last_query != query_aktif or btn_cari:
                        
                        # 1. Logika Morphology
                        if "Morphology" in mode_pencarian:
                            with st.spinner(f"Menganalisis variasi morfologi dari '{query_aktif}'..."):
                                try:
                                    target_lemma = nlp(query_aktif)[0].lemma_.lower()
                                    morph_data = []
                                    for fname in selected_files: 
                                        teks_mentah = str(st.session_state.local_files[fname]['cleaned'])
                                        doc_morph = nlp(teks_mentah[:150000])
                                        for token in doc_morph:
                                            if token.lemma_.lower() == target_lemma and not token.is_punct and not token.is_space:
                                                fitur_morfologi = str(token.morph)
                                                morph_data.append({
                                                    "Bentuk Teks Asli": token.text.lower(),
                                                    "Kelas Kata (POS)": token.pos_,
                                                    "Struktur Morfologi": fitur_morfologi if fitur_morfologi else "Bentuk Dasar (Base)"
                                                })
                                    
                                    if morph_data:
                                        df_morph = pd.DataFrame(morph_data)
                                        df_grouped = df_morph.groupby(['Bentuk Teks Asli', 'Kelas Kata (POS)', 'Struktur Morfologi']).size().reset_index(name='Frekuensi Muncul')
                                        st.session_state.morph_results = df_grouped.sort_values('Frekuensi Muncul', ascending=False).reset_index(drop=True)
                                    else:
                                        st.session_state.morph_results = pd.DataFrame()
                                    
                                    st.session_state.search_results = [] 
                                    st.session_state.sketch_results = {}
                                    st.session_state.last_query = query_aktif
                                except Exception as e: st.error(f"Error: {e}")
                                
                        # 2. Logika Word Sketch
                        elif "Word Sketch" in mode_pencarian:
                            with st.spinner(f"Membangun profil tata bahasa untuk '{query_aktif}'..."):
                                try:
                                    t_gabung = " ".join([st.session_state.local_files[f]['cleaned'] for f in selected_files])
                                    h_ws = buat_word_sketch(query_aktif, t_gabung)
                                    
                                    st.session_state.sketch_results = h_ws
                                    st.session_state.search_results = []
                                    st.session_state.morph_results = pd.DataFrame()
                                    st.session_state.last_query = query_aktif
                                except Exception as e: st.error(f"Error: {e}")
                        
                        # 3. Logika Normal Search
                        else:
                            with st.spinner("Mencari & Memproses Struktur Kalimat (Harap Tunggu)..."):
                                matches_global = []
                                if "Lemmatization" in mode_pencarian or "Semantic" in mode_pencarian: query_doc = nlp(query_aktif)
                                
                                for fname in selected_files: 
                                    teks_b = st.session_state.local_files[fname]['cleaned']
                                    doc = nlp(teks_b[:100000])
                                    
                                    if "POS Search" in mode_pencarian:
                                        for s in doc.sents:
                                            match_found, highlighted = False, ""
                                            for token in s:
                                                match_kw = (token.lemma_.lower() == pos_keyword) if pos_keyword else True
                                                match_pos = (token.pos_ == target_pos_tag) if target_pos_tag != "ALL" else True
                                                if match_kw and match_pos and (pos_keyword or target_pos_tag != "ALL"):
                                                    match_found = True
                                                    highlighted += f"<mark style='background:#0EA5E9; color:white; font-weight:bold; padding:0 4px; border-radius:4px;'>{token.text}</mark>{token.whitespace_}"
                                                else: highlighted += f"{token.text}{token.whitespace_}"
                                            if match_found:
                                                p_counts = Counter([t.pos_ for t in s if t.pos_ not in ['SPACE', 'PUNCT', 'SYM', 'X']])
                                                p_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>" + "".join([f"<span style='background-color: {Warna_POS_Utama.get(p, '#94A3B8')}; color:white; border-radius:4px; padding:3px 8px; font-size:11px; font-weight:600;'>{p}: {c}</span>" for p, c in p_counts.items()]) + "</div>"
                                                matches_global.append({'file': fname, 'text': s.text.strip(), 'html': highlighted, 'pills': p_html, 'tags': list(set([t.pos_ for t in s if t.pos_ in deskripsi_pos]))})
                                                
                                    elif "NER" in mode_pencarian:
                                        target_ent = query_aktif.split(" ")[0]
                                        for s in doc.sents:
                                            if any(ent.label_ == target_ent for ent in s.ents):
                                                m_text = s.text.strip()
                                                h_light = get_colored_ner_inline(m_text, target_ent)
                                                p_counts = Counter([t.pos_ for t in s if t.pos_ not in ['SPACE', 'PUNCT', 'SYM', 'X']])
                                                p_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>" + "".join([f"<span style='background-color: {Warna_POS_Utama.get(p, '#94A3B8')}; color:white; border-radius:4px; padding:3px 8px; font-size:11px; font-weight:600;'>{p}: {c}</span>" for p, c in p_counts.items()]) + "</div>"
                                                matches_global.append({'file': fname, 'text': m_text, 'html': h_light, 'pills': p_html, 'tags': list(set([t.pos_ for t in s if t.pos_ in deskripsi_pos]))})
                                    
                                    elif "Dependency" in mode_pencarian:
                                        if head_word != "" or child_word != "":
                                            for s in doc.sents:
                                                match_found, highlighted = False, s.text.strip()
                                                for token in s:
                                                    if token.dep_ == rel_type or (rel_type == "obj" and token.dep_ == "dobj"):
                                                        if (head_word == "" or token.head.lemma_.lower() == head_word) and (child_word == "" or token.lemma_.lower() == child_word):
                                                            match_found = True
                                                            if head_word != "": highlighted = re.sub(f"\\b({token.head.text})\\b", r"<mark style='background:#F59E0B; color:white; padding:0 4px; border-radius:3px;'>\1</mark>", highlighted, flags=re.I)
                                                            if child_word != "": highlighted = re.sub(f"\\b({token.text})\\b", r"<mark style='background:#0EA5E9; color:white; padding:0 4px; border-radius:3px;'>\1</mark>", highlighted, flags=re.I)
                                                            break
                                                if match_found:
                                                    p_counts = Counter([t.pos_ for t in s if t.pos_ not in ['SPACE', 'PUNCT', 'SYM', 'X']])
                                                    p_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>" + "".join([f"<span style='background-color: {Warna_POS_Utama.get(p, '#94A3B8')}; color:white; border-radius:4px; padding:3px 8px; font-size:11px; font-weight:600;'>{p}: {c}</span>" for p, c in p_counts.items()]) + "</div>"
                                                    matches_global.append({'file': fname, 'text': s.text.strip(), 'html': highlighted, 'pills': p_html, 'tags': list(set([t.pos_ for t in s if t.pos_ in deskripsi_pos]))})
                                    
                                    elif "Regex" in mode_pencarian:
                                        try:
                                            pola_regex = re.compile(query_aktif)
                                            for s in doc.sents:
                                                m_text = s.text.strip()
                                                if pola_regex.search(m_text):
                                                    h_light = pola_regex.sub(r"<mark style='background:#F59E0B; color:white; padding:0 4px; border-radius:3px;'>\g<0></mark>", m_text)
                                                    p_counts = Counter([t.pos_ for t in s if t.pos_ not in ['SPACE', 'PUNCT', 'SYM', 'X']])
                                                    p_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>" + "".join([f"<span style='background-color: {Warna_POS_Utama.get(p, '#94A3B8')}; color:white; border-radius:4px; padding:3px 8px; font-size:11px; font-weight:600;'>{p}: {c}</span>" for p, c in p_counts.items()]) + "</div>"
                                                    matches_global.append({'file': fname, 'text': m_text, 'html': h_light, 'pills': p_html, 'tags': list(set([t.pos_ for t in s if t.pos_ in deskripsi_pos]))})
                                        except: pass

                                    elif "Lemmatization" in mode_pencarian:
                                        q_l = query_doc[0].lemma_.lower() if len(query_doc) > 0 else query_aktif.lower()
                                        for s in doc.sents:
                                            if any(t.lemma_.lower() == q_l for t in s):
                                                m_text = s.text.strip()
                                                k_c = set([t.text for t in s if t.lemma_.lower() == q_l])
                                                h_light = re.sub(r"\b(" + "|".join(map(re.escape, k_c)) + r")\b", r"<mark style='background:#0EA5E9; color:white; padding:0 4px; border-radius:3px;'>\1</mark>", m_text, flags=re.I) if k_c else m_text
                                                p_counts = Counter([t.pos_ for t in s if t.pos_ not in ['SPACE', 'PUNCT', 'SYM', 'X']])
                                                p_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>" + "".join([f"<span style='background-color: {Warna_POS_Utama.get(p, '#94A3B8')}; color:white; border-radius:4px; padding:3px 8px; font-size:11px; font-weight:600;'>{p}: {c}</span>" for p, c in p_counts.items()]) + "</div>"
                                                matches_global.append({'file': fname, 'text': m_text, 'html': h_light, 'pills': p_html, 'tags': list(set([t.pos_ for t in s if t.pos_ in deskripsi_pos]))})
                                    
                                    elif "Semantic" in mode_pencarian:
                                        if len(query_doc) > 0 and query_doc[0].has_vector:
                                            for s in doc.sents:
                                                if len(s.text.strip()) > 5:
                                                    is_sim = False
                                                    for t in s:
                                                        if not t.is_stop and not t.is_punct and t.has_vector:
                                                            if query_doc.similarity(t) >= 0.60: is_sim = True; break 
                                                    if is_sim:
                                                        p_counts = Counter([t.pos_ for t in s if t.pos_ not in ['SPACE', 'PUNCT', 'SYM', 'X']])
                                                        p_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>" + "".join([f"<span style='background-color: {Warna_POS_Utama.get(p, '#94A3B8')}; color:white; border-radius:4px; padding:3px 8px; font-size:11px; font-weight:600;'>{p}: {c}</span>" for p, c in p_counts.items()]) + "</div>"
                                                        matches_global.append({'file': fname, 'text': s.text.strip(), 'html': s.text.strip(), 'pills': p_html, 'tags': list(set([t.pos_ for t in s if t.pos_ in deskripsi_pos]))})

                                    elif "Boolean" in mode_pencarian:
                                        q_r = query_aktif.replace("AND", "&").replace("OR", "|").replace("NOT", "!")
                                        for s in doc.sents:
                                            lems = {t.lemma_.lower() for t in s}
                                            m_f = False
                                            if " & " in q_r: m_f = all(nlp(p.strip())[0].lemma_.lower() in lems for p in q_r.split("&"))
                                            elif " | " in q_r: m_f = any(nlp(p.strip())[0].lemma_.lower() in lems for p in q_r.split("|"))
                                            elif "!" in q_r: m_f = nlp(q_r.replace("!", "").strip())[0].lemma_.lower() not in lems
                                            else: m_f = query_aktif.lower() in lems

                                            if m_f:
                                                h_light = s.text.strip()
                                                for w in re.findall(r'\w+', query_aktif):
                                                    if w.upper() not in ["AND", "OR", "NOT"]: h_light = re.sub(f"\\b({w})\\b", r"<mark style='background:#FDE047; color:#0F172A; padding:0 4px; border-radius:3px;'>\1</mark>", h_light, flags=re.I)
                                                p_counts = Counter([t.pos_ for t in s if t.pos_ not in ['SPACE', 'PUNCT', 'SYM', 'X']])
                                                p_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>" + "".join([f"<span style='background-color: {Warna_POS_Utama.get(p, '#94A3B8')}; color:white; border-radius:4px; padding:3px 8px; font-size:11px; font-weight:600;'>{p}: {c}</span>" for p, c in p_counts.items()]) + "</div>"
                                                matches_global.append({'file': fname, 'text': s.text.strip(), 'html': h_light, 'pills': p_html, 'tags': list(set([t.pos_ for t in s if t.pos_ in deskripsi_pos]))})

                                st.session_state.search_results = matches_global
                                st.session_state.sketch_results = {}
                                st.session_state.morph_results = pd.DataFrame()
                                st.session_state.current_page = 0
                                st.session_state.last_query = query_aktif

                # ==========================================
                # TAMPILAN HASIL (BERDASARKAN MODE)
                # ==========================================
                if "Word Sketch" in mode_pencarian:
                    if st.session_state.sketch_results:
                        h_ws = st.session_state.sketch_results
                        st.write(f"### 🔍 Hasil Profil untuk: **{query_aktif}**")
                        c_mod, c_subj, c_obj = st.columns(3)
                        
                        def render_clickable_list(header, data, color, kp):
                            st.markdown(f"<div style='background:{color}; padding:10px; border-radius:8px; margin-bottom:10px;'><b>{header}</b></div>", unsafe_allow_html=True)
                            if data:
                                for i, (k, jml) in enumerate(data):
                                    st.button(f"{k} ({jml}x)", key=f"{kp}_{query_aktif}_{k}_{i}", use_container_width=True, on_click=update_ws_word, args=(k,))
                            else: st.caption("Tidak ditemukan.")

                        with c_mod: render_clickable_list("✨ Modifiers (Sifat)", h_ws["✨ Modifiers (Sifat/Penjelas)"], "#F0FDF4", "ws_mod")
                        with c_subj: render_clickable_list("🏃‍♂️ Sebagai Pelaku", h_ws["🏃‍♂️ Sebagai Subjek (Melakukan)"], "#EFF6FF", "ws_sub")
                        with c_obj: render_clickable_list("🎯 Sebagai Korban", h_ws["🎯 Sebagai Objek (Dikenai)"], "#FEF2F2", "ws_obj")
                    elif query_aktif and st.session_state.last_query == query_aktif:
                        st.warning(f"🔍 Profil kata '{query_aktif}' tidak ditemukan.")
                        
                elif "Morphology" in mode_pencarian:
                    if not st.session_state.morph_results.empty:
                        df_grouped = st.session_state.morph_results
                        st.success(f"✅ Ditemukan **{len(df_grouped)}** variasi bentuk kata untuk akar kata '{query_aktif}'.")
                        
                        ch_morph = alt.Chart(df_grouped).mark_bar(color="#F59E0B", cornerRadiusEnd=4).encode(
                            x=alt.X('Frekuensi Muncul:Q', title='Jumlah Muncul di Dokumen'),
                            y=alt.Y('Bentuk Teks Asli:N', sort='-x', title='Variasi Kata Teks'),
                            tooltip=['Bentuk Teks Asli', 'Kelas Kata (POS)', 'Struktur Morfologi', 'Frekuensi Muncul']
                        ).properties(height=300).configure_view(stroke='#94A3B8', strokeWidth=1)
                        st.altair_chart(ch_morph, use_container_width=True)
                        
                        st.markdown("<div style='font-weight:bold; margin-bottom:10px;'>Tabel Rincian Morfologi (Bisa di-Sortir):</div>", unsafe_allow_html=True)
                        st.dataframe(df_grouped, use_container_width=True, hide_index=True)
                    elif query_aktif and st.session_state.last_query == query_aktif:
                        st.warning(f"🔍 Akar kata '{query_aktif}' tidak digunakan di dalam dokumen ini.")
                
                else:
                    if not st.session_state.search_results and query_aktif and st.session_state.last_query == query_aktif:
                        st.warning(f"🔍 Pencarian '{query_aktif}' tidak ditemukan.")

                    if st.session_state.search_results:
                        tot_res = len(st.session_state.search_results)
                        c_i, _, c_ft, c_fd = st.columns([5, 1, 3, 1.05], gap="small")
                        c_i.markdown(f"<div style='color:#059669; font-weight:bold; padding-top:8px;'>✅ Ditemukan {tot_res} kalimat.</div>", unsafe_allow_html=True)
                        c_ft.markdown("<div style='text-align:right; padding-top:8px;'>Tampilkan per halaman:</div>", unsafe_allow_html=True)
                        limit_opt = [str(x) for x in [5, 10, 25, 50] if x < tot_res] + [f"All ({tot_res})"]
                        limit_sel = c_fd.selectbox("Limit", limit_opt, label_visibility="collapsed")
                        
                        IP_PAGE = tot_res if limit_sel.startswith("All") else int(limit_sel)
                        tot_pages = max(1, math.ceil(tot_res / IP_PAGE))
                        if st.session_state.current_page >= tot_pages: st.session_state.current_page = max(0, tot_pages - 1)
                        
                        s_idx = st.session_state.current_page * IP_PAGE
                        for i, m_data in enumerate(st.session_state.search_results[s_idx:s_idx+IP_PAGE]):
                            with st.container(border=True):
                                st.markdown(f"""
                                <div style='color:#334155; font-size:15.5px; margin-bottom:15px; line-height:1.6;'>{m_data['html']}</div>
                                <div style='display:flex; justify-content:space-between; align-items:center; border-top:1px solid #F1F5F9; padding-top:12px;'>
                                    <div style='display:flex; align-items:center; gap:12px; flex-wrap:wrap;'>
                                        <div style='font-size:11px; color:#0284C7; font-weight:bold; background:#F0F9FF; padding:5px 10px; border-radius:4px; border:1px solid #BAE6FD;'>📄 {m_data['file'].upper()}</div>
                                        <div>{m_data['pills']}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                _, c_ak = st.columns([8, 2])
                                aksi = c_ak.selectbox("Aksi", ["Aksi", "🏷️ POS Tag", "🌿 Syntax Tree", "🌐 Trans"], key=f"aksi_{s_idx+i}", label_visibility="collapsed")

                                if aksi == "🏷️ POS Tag":
                                    st.markdown(get_colored_pos_text(m_data['text']), unsafe_allow_html=True)
                                    o_drop = [t for t in deskripsi_pos.keys() if t in m_data['tags']]
                                    if o_drop: st.info(deskripsi_pos[st.selectbox("💡 Penjelasan label:", o_drop, key=f"hp_{s_idx+i}")])
                                elif aksi == "🌿 Syntax Tree":
                                    with st.spinner("Membedah struktur..."): st.markdown(render_dependency_tree(m_data['text']), unsafe_allow_html=True)
                                elif aksi == "🌐 Trans":
                                    c_l, c_g = st.columns([7, 3])
                                    tl_name = c_l.selectbox("Ke:", list(DAFTAR_BAHASA.keys()), key=f"ln_{s_idx+i}", label_visibility="collapsed")
                                    if c_g.button("Ok", key=f"g_{s_idx+i}", use_container_width=True):
                                        with st.spinner("Menerjemahkan..."):
                                            try: st.markdown(f"<div style='background:#E0F2FE; border:1px solid #7DD3FC; padding:15px; border-radius:8px; margin-top:10px;'>{GoogleTranslator(source='auto', target=DAFTAR_BAHASA[tl_name]).translate(m_data['text'])}</div>", unsafe_allow_html=True)
                                            except Exception as e: st.error(f"Error: {e}")

                        st.write("") 
                        c_b1, c_p, c_pi, c_n, c_b2 = st.columns([2, 1, 2, 1, 2])
                        c_p.button("⬅️ Prev", use_container_width=True, disabled=(st.session_state.current_page == 0), on_click=prev_page_1)
                        c_pi.markdown(f"<div style='text-align:center; padding-top:8px;'>Halaman: <b>{st.session_state.current_page + 1} / {tot_pages}</b></div>", unsafe_allow_html=True)
                        c_n.button("Next ➡️", use_container_width=True, disabled=(st.session_state.current_page >= tot_pages - 1), on_click=next_page_1)

            # ==========================================
            # TAB 3: ANALISIS MENDALAM 
            # ==========================================
            with tab_summary:
                st.markdown("<h3 style='color:#0F172A;'>📝 Analisis Dokumen Mendalam</h3>", unsafe_allow_html=True)
                
                target_file = st.selectbox("Pilih dokumen spesifik untuk dianalisis:", selected_files, key="sel_doc_sum_sent_top")
                sub_topic, sub_sum, sub_sent = st.tabs(["🗂️ Pemodelan Topik (LDA)", "📑 Ekstraksi Ringkasan", "😊 Analisis Sentimen (VADER)"])
                
                # --- SUB TAB: PEMODELAN TOPIK ---
                with sub_topic:
                    with st.expander("📖 Panduan: Cara Membaca Topik"):
                        st.info("""
                        **Fungsi:** Algoritma AI (Latent Dirichlet Allocation) membaca dokumen Anda dan secara otomatis mengelompokkan kata-kata ke dalam beberapa "Tema/Topik Utama" tanpa bantuan manusia.
                        
                        **Tips Membaca:**
                        * Tiap kotak mewakili satu gagasan atau konteks bahasan utama dalam dokumen.
                        * Jika kata-kata di antara topik terlihat mirip/tumpang tindih, coba **kurangi Jumlah Topik**.
                        * Fitur ini sangat cocok untuk membedah jurnal, artikel berita, atau laporan penelitian panjang.
                        """)

                    col_t1, col_t2 = st.columns([4, 2], vertical_alignment="bottom")
                    with col_t1: num_topics = st.number_input("Tentukan Jumlah Topik Utama:", min_value=2, max_value=12, value=3, key="num_topics")
                    with col_t2: btn_topic = st.button("🚀 Ekstrak Topik Otomatis", type="primary", use_container_width=True)

                    if btn_topic:
                        with st.spinner(f"Membangun model kecerdasan buatan (LDA) untuk '{target_file}'..."):
                            from sklearn.feature_extraction.text import CountVectorizer
                            from sklearn.decomposition import LatentDirichletAllocation
                            
                            teks_mentah = st.session_state.local_files[target_file]['cleaned']
                            doc_topic = nlp(teks_mentah[:150000])
                            
                            corpus_sentences = [s.text.lower() for s in doc_topic.sents if len(s.text.split()) > 4]
                            
                            if len(corpus_sentences) < 10:
                                st.warning("Teks terlalu pendek. Butuh minimal 10 kalimat untuk membangun model topik.")
                            else:
                                try:
                                    custom_stops = {'et', 'al', 'fig', 'figure', 'table', 'use', 'used', 'using', 'based', 'study', 'result', 'results', 'analysis'}
                                    stopwords_gabungan = list(nlp.Defaults.stop_words.union(custom_stops))
                                    
                                    vectorizer = CountVectorizer(stop_words=stopwords_gabungan, min_df=2, max_df=0.9, token_pattern=r'\b[a-z]{3,}\b')
                                    X = vectorizer.fit_transform(corpus_sentences)
                                    
                                    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, max_iter=10)
                                    lda.fit(X)
                                    
                                    feature_names = vectorizer.get_feature_names_out()
                                    st.success("✅ Model topik berhasil dibangun!")
                                    st.write("")
                                    
                                    cols_topik = st.columns(3) 
                                    for topic_idx, topic in enumerate(lda.components_):
                                        col_idx = topic_idx % 3
                                        top_words_idx = topic.argsort()[:-11:-1] 
                                        top_words = [feature_names[i] for i in top_words_idx]
                                        
                                        with cols_topik[col_idx]:
                                            st.markdown(f"""
                                            <div style='background:#F8FAFC; border:1px solid #CBD5E1; padding:20px; border-radius:12px; margin-bottom:20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);'>
                                                <h4 style='color:#0369A1; margin-top:0; border-bottom:2px solid #BAE6FD; padding-bottom:8px; font-size:16px;'>📌 Topik {topic_idx + 1}</h4>
                                                <div style='display:flex; flex-wrap:wrap; gap:6px; margin-top:15px;'>
                                                    {''.join([f"<span style='background:white; border:1px solid #94A3B8; border-radius:12px; padding:4px 10px; font-size:13.5px; color:#334155; font-weight:500;'>{w}</span>" for w in top_words])}
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                except Exception as e: st.error(f"Gagal mengekstrak topik. Detail: {e}")

                # --- SUB TAB: RINGKASAN ---
                with sub_sum:
                    with st.expander("📖 Panduan: Cara Menggunakan Summarization"):
                        st.info("""
                        **Deskripsi:** Menggunakan algoritma *LexRank* untuk membaca seluruh isi dokumen Anda dan mengekstrak kalimat-kalimat yang paling krusial dan mewakili seluruh isi teks.
                        """)
                        
                    if st.button(f"🚀 Mulai Ekstraksi Ringkasan", type="primary", key="btn_run_sum"):
                        with st.spinner("Mengekstrak informasi penting..."):
                            try:
                                t_target = st.session_state.local_files[target_file]['text']
                                t_kalimat = st.session_state.local_files[target_file]['stats']['k']
                                t_jml = max(5, min(int(t_kalimat * 0.15), 30))
                                
                                parser = PlaintextParser.from_string(t_target, Tokenizer("english"))
                                h_eks = LexRankSummarizer()(parser.document, t_jml)
                                
                                st.session_state.summary_results[target_file] = "\n\n".join([str(s) for s in h_eks])
                                st.success("✅ Ekstraksi selesai!")
                            except Exception as e: st.error(f"Error: {e}")

                    if target_file in st.session_state.summary_results:
                        t_hasil = st.session_state.summary_results[target_file]
                        with st.container(border=True):
                            cj, ca = st.columns([0.9, 0.1])
                            with cj: st.markdown(f"#### 📑 Teks Ringkasan")
                            with ca:
                                with st.popover("⋮"):
                                    st.download_button("📄 TXT", data=t_hasil.encode('utf-8'), file_name=f"sum_{target_file}.txt", use_container_width=True)
                                    try:
                                        doc_ex = Document(); doc_ex.add_heading(f"Ringkasan", 0)
                                        for p in t_hasil.split('\n\n'):
                                            if p.strip(): doc_ex.add_paragraph(p.strip())
                                        b_docx = io.BytesIO(); doc_ex.save(b_docx)
                                        st.download_button("📝 DOCX", data=b_docx.getvalue(), file_name=f"sum_{target_file}.docx", use_container_width=True)
                                    except: pass

                            st.markdown("<hr style='margin: 10px 0; opacity: 0.2;'>", unsafe_allow_html=True)
                            st.markdown(f"<div style='color: #334155; font-size: 16px; line-height: 1.8; text-align: justify;'>{t_hasil.replace(chr(10)+chr(10), '<br><br>')}</div>", unsafe_allow_html=True)
                        
                        st.write("")
                        ak_sum = st.selectbox("Analisis Lanjutan:", ["Pilih Aksi...", "🏷️ POS Tagging", "🌐 Translate"], key="ak_s_l")
                        if ak_sum == "🏷️ POS Tagging":
                            with st.spinner("Membedah..."): st.markdown(get_colored_pos_text(t_hasil), unsafe_allow_html=True)
                        elif ak_sum == "🌐 Translate":
                            cls, cgs, _ = st.columns([2, 1, 5])
                            with cls: ts_code = DAFTAR_BAHASA[st.selectbox("Ke:", list(DAFTAR_BAHASA.keys()), key="l_s_u", label_visibility="collapsed")]
                            with cgs: 
                                if st.button("Ok", key="g_s_u", use_container_width=True):
                                    with st.spinner("Translating..."):
                                        try: st.markdown(f"<div style='background-color:#F0FDF4; padding:25px; border-radius:12px; margin-top:15px;'>{GoogleTranslator(source='auto', target=ts_code).translate(t_hasil[:4500])}</div>", unsafe_allow_html=True)
                                        except: st.error("Gagal terjemahkan.")

                # --- SUB TAB: SENTIMEN ---
                with sub_sent:
                    with st.expander("📖 Panduan: Cara Membaca Sentimen"):
                        st.info("""
                        **Fungsi:** Mengukur nada emosi dari sebuah dokumen (Positif, Negatif, atau Netral) dengan membedahnya kalimat per kalimat.
                        
                        **Cara Membaca Skor:**
                        * **Skor Compound:** Total sentimen keseluruhan dokumen. Bernilai antara **-1.0 (Sangat Negatif)** hingga **+1.0 (Sangat Positif)**.
                        * Daftar di bawah akan memisahkan 5 kalimat paling positif dan 5 kalimat paling negatif.
                        """)
                    
                    if st.button(f"🎭 Mulai Analisis Sentimen", type="primary", key="btn_run_sent"):
                        from nltk.sentiment import SentimentIntensityAnalyzer
                        with st.spinner(f"Menganalisis emosi kalimat di '{target_file}'..."):
                            teks_mentah = st.session_state.local_files[target_file]['cleaned']
                            doc_sent = nlp(teks_mentah[:100000])
                            
                            sia = SentimentIntensityAnalyzer()
                            data_sentimen = []
                            
                            for s in doc_sent.sents:
                                teks_kalimat = s.text.strip()
                                if len(teks_kalimat) > 15:
                                    scores = sia.polarity_scores(teks_kalimat)
                                    data_sentimen.append({
                                        "Kalimat": teks_kalimat,
                                        "Positif": scores['pos'],
                                        "Negatif": scores['neg'],
                                        "Netral": scores['neu'],
                                        "Compound": scores['compound']
                                    })
                            
                            if data_sentimen:
                                df_sent = pd.DataFrame(data_sentimen)
                                avg_compound = df_sent['Compound'].mean()
                                
                                if avg_compound >= 0.05:
                                    status, bg_color, txt_color = "Positif 😊", "#DCFCE7", "#166534"
                                elif avg_compound <= -0.05:
                                    status, bg_color, txt_color = "Negatif 😠", "#FEE2E2", "#991B1B"
                                else:
                                    status, bg_color, txt_color = "Netral 😐", "#F1F5F9", "#334155"
                                    
                                st.markdown(f"""
                                <div style='background-color: {bg_color}; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px; border: 1px solid {txt_color}44;'>
                                    <h4 style='color: {txt_color}; margin: 0; font-size: 18px;'>Sentimen Keseluruhan Dokumen</h4>
                                    <h1 style='color: {txt_color}; margin: 10px 0 0 0; font-size: 42px;'>{status}</h1>
                                    <p style='color: {txt_color}; margin: 5px 0 0 0; font-size: 16px;'>Skor Rata-rata: <b>{avg_compound:.3f}</b></p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.write("#### 📊 Distribusi Sentimen Kalimat")
                                df_sent['Kategori'] = df_sent['Compound'].apply(lambda c: 'Positif' if c >= 0.05 else ('Negatif' if c <= -0.05 else 'Netral'))
                                distribusi = df_sent['Kategori'].value_counts().reset_index()
                                distribusi.columns = ['Kategori', 'Jumlah Kalimat']
                                
                                chart_dist = alt.Chart(distribusi).mark_arc(innerRadius=50).encode(
                                    theta=alt.Theta(field="Jumlah Kalimat", type="quantitative"),
                                    color=alt.Color(field="Kategori", type="nominal", scale=alt.Scale(domain=['Positif', 'Netral', 'Negatif'], range=['#10B981', '#94A3B8', '#EF4444'])),
                                    tooltip=['Kategori', 'Jumlah Kalimat']
                                ).properties(height=300)
                                
                                col_pie, col_ket = st.columns([1, 1])
                                with col_pie: st.altair_chart(chart_dist, use_container_width=True)
                                with col_ket: st.dataframe(distribusi, use_container_width=True, hide_index=True)

                                st.markdown("<hr style='border: 1px dashed #E2E8F0; margin: 20px 0;'>", unsafe_allow_html=True)
                                
                                col_pos, col_neg = st.columns(2)
                                with col_pos:
                                    st.markdown("<h4 style='color:#10B981;'>📈 5 Kalimat Paling Positif</h4>", unsafe_allow_html=True)
                                    for _, row in df_sent.nlargest(5, 'Compound').iterrows():
                                        st.markdown(f"<div style='background:#ECFDF5; padding:10px; border-radius:6px; border-left:4px solid #10B981; margin-bottom:8px; font-size:14px; color:#064E3B; line-height:1.5;'>{row['Kalimat']} <br><small><b>Skor: +{row['Compound']:.2f}</b></small></div>", unsafe_allow_html=True)
                                with col_neg:
                                    st.markdown("<h4 style='color:#EF4444;'>📉 5 Kalimat Paling Negatif</h4>", unsafe_allow_html=True)
                                    for _, row in df_sent.nsmallest(5, 'Compound').iterrows():
                                        st.markdown(f"<div style='background:#FEF2F2; padding:10px; border-radius:6px; border-left:4px solid #EF4444; margin-bottom:8px; font-size:14px; color:#7F1D1D; line-height:1.5;'>{row['Kalimat']} <br><small><b>Skor: {row['Compound']:.2f}</b></small></div>", unsafe_allow_html=True)
                            else: st.warning("Tidak ada kalimat yang bisa dianalisis.")

        fitur_nlp_dashboard(selected_files)
else:
    st.info("👋 Silakan upload file terlebih dahulu untuk mulai menggunakan dashboard.")

