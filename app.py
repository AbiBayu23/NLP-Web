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
import time

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

@st.cache_resource
def siapkan_kamus_sinonim():
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
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

# Gunakan session_state untuk mengontrol key uploader agar bisa di-reset
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_files = st.file_uploader(
    "Upload File (PDF, DOCX, TXT)", 
    accept_multiple_files=True, 
    type=['pdf', 'docx', 'txt'],
    key=f"uploader_{st.session_state.uploader_key}" # Key dinamis
)

# Memproses file baru
if uploaded_files:
    ada_file_baru = False
    for file in uploaded_files:
        if file.name not in st.session_state.local_files:
            with st.spinner(f"Menganalisis {file.name}..."):
                raw_text = extract_text(file)
                teks_bersih = bersihkan_teks_untuk_analisis(raw_text)
                pola_kata = r'\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b'
                semua_kata = re.findall(pola_kata, raw_text.lower())
                
                # Simpan ke memori sistem (local_files)
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

    # TRIK PENTING: Jika ada file baru selesai diproses, 
    # kita ubah key uploader agar notifikasi upload-nya menghilang secara instan.
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
                # Mencari file ini masuk ke grup mana saja
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
            
            # Tombol Hapus Terpilih
            file_untuk_dihapus = edited_df[edited_df["Hapus"] == True]["Nama File"].tolist()
            if st.button("🗑️ Hapus File Terpilih", type="primary", disabled=not file_untuk_dihapus):
                for fname in file_untuk_dihapus:
                    del st.session_state.local_files[fname]
                    # Hapus juga dari semua sub-corpora
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
        # 5. TAB FITUR NLP 
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
            def prev_page_2(): st.session_state.ps_current_page -= 1
            def next_page_2(): st.session_state.ps_current_page += 1

            # Tambahkan tab "🧩 Word Sketch" ke dalam daftar
            tab_vis, tab_compare, tab_search, tab_sketch, tab_ngram, tab_summary = st.tabs([
                "📊 Visualisasi", "⚖️ Perbandingan", "🔍 Pencarian Pintar", "🧩 Word Sketch", "🔢 N-Grams", "📝 Ringkasan"
            ])
            with tab_vis:
                st.markdown(f"<h3 style='color:#0F172A;'>📊 Gambaran Visual Dokumen</h3>", unsafe_allow_html=True)

                st.markdown("""
                    <style>
                    /* Menambahkan [data-testid="stExpander"] agar efek slider HANYA berlaku di dalam expander visualisasi */
                    [data-testid="stExpander"] [data-testid="stHorizontalBlock"] {
                        flex-wrap: nowrap !important;
                        overflow-x: auto !important; 
                        padding-bottom: 15px;
                    }
                    [data-testid="stExpander"] [data-testid="stHorizontalBlock"] > div {
                        min-width: 600px !important;
                        width: 800px !important;
                        flex: 0 0 auto !important;
                    }
                    </style>
                """, unsafe_allow_html=True)

                with st.expander("Klik di sini untuk Membuka / Menutup Visualisasi Data", expanded=False):
                    with st.container():
                        for fname in selected_files:
                            st.markdown(f"#### 📄 Laporan: {fname}")
                            
                            # --- MULAI BLOK CACHING DATA ---
                            if 'vis_cache' not in st.session_state.local_files[fname]:
                                with st.spinner(f"Menyiapkan visualisasi, Concordance, dan N-Grams untuk {fname}..."):
                                    teks_dokumen = st.session_state.local_files[fname]['text']
                                    df_pos, df_words, df_cloud_text = dapatkan_data_visual(teks_dokumen[:80000])
                                    
                                    teks_mentah_aktif = st.session_state.local_files[fname]['cleaned']
                                    
                                    # 1. Hitung Concordance (Kolokasi)
                                    if not df_words.empty and 'Pasangan 1' not in df_words.columns:
                                        col1, col2, col3, col4, col5 = [], [], [], [], []
                                        for i, kata in enumerate(df_words['Kata']):
                                            if i < 500:  
                                                hasil_colloc = hitung_collocation(kata, teks_mentah_aktif, window=5)
                                                col1.append(f"{hasil_colloc[0][0]} ({hasil_colloc[0][1]}x)" if len(hasil_colloc) > 0 else "-")
                                                col2.append(f"{hasil_colloc[1][0]} ({hasil_colloc[1][1]}x)" if len(hasil_colloc) > 1 else "-")
                                                col3.append(f"{hasil_colloc[2][0]} ({hasil_colloc[2][1]}x)" if len(hasil_colloc) > 2 else "-")
                                                col4.append(f"{hasil_colloc[3][0]} ({hasil_colloc[3][1]}x)" if len(hasil_colloc) > 3 else "-")
                                                col5.append(f"{hasil_colloc[4][0]} ({hasil_colloc[4][1]}x)" if len(hasil_colloc) > 4 else "-")
                                            else:
                                                col1.append("-"); col2.append("-"); col3.append("-"); col4.append("-"); col5.append("-")
                                        df_words['Pasangan 1'] = col1; df_words['Pasangan 2'] = col2; df_words['Pasangan 3'] = col3; df_words['Pasangan 4'] = col4; df_words['Pasangan 5'] = col5

                                    # 2. Ekstrak N-Grams (Bigram & Trigram)
                                    words_ng = re.findall(r'\b[a-z]{3,}\b', teks_mentah_aktif.lower())
                                    stop_words_ng = nlp.Defaults.stop_words
                                    words_ng = [w for w in words_ng if w not in stop_words_ng]
                                    
                                    bigrams = [" ".join(g) for g in zip(*[words_ng[i:] for i in range(2)])]
                                    trigrams = [" ".join(g) for g in zip(*[words_ng[i:] for i in range(3)])]
                                    
                                    df_bigram = pd.DataFrame(Counter(bigrams).most_common(30), columns=['Frasa', 'Frekuensi'])
                                    df_trigram = pd.DataFrame(Counter(trigrams).most_common(30), columns=['Frasa', 'Frekuensi'])

                                    # 3. Word Cloud
                                    cloud_img = None
                                    if df_cloud_text:
                                        fig = get_cached_wordcloud(df_cloud_text[:80000])
                                        buf = io.BytesIO()
                                        fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
                                        plt.close(fig)
                                        cloud_img = buf.getvalue()

                                    st.session_state.local_files[fname]['vis_cache'] = {
                                        'df_pos': df_pos,
                                        'df_words': df_words,
                                        'df_bigram': df_bigram,
                                        'df_trigram': df_trigram,
                                        'cloud_img': cloud_img
                                    }
                            # --- SELESAI BLOK CACHING DATA ---

                            # Ambil data dari memory
                            cache_vis = st.session_state.local_files[fname]['vis_cache']
                            
                            # BARIS 1: Grafik POS, Kata, dan WordCloud
                            col_chart1, col_chart2, col_chart3 = st.columns(3)
                            with col_chart1:
                                st.caption(f"Statistik Tata Bahasa (**{fname}**)")
                                if not cache_vis['df_pos'].empty:
                                    chart_pos = alt.Chart(cache_vis['df_pos']).mark_bar(color="#0284C7", cornerRadiusEnd=4).encode(
                                        y=alt.Y('POS Tag', sort='-x', title='Kelas Kata'), 
                                        x=alt.X('Jumlah Kata', title='Total Jumlah'),
                                        tooltip=['POS Tag', 'Jumlah Kata']
                                    ).properties(height=380, width=800).configure_view(stroke='#94A3B8', strokeWidth=1)
                                    st.altair_chart(chart_pos, use_container_width=True)
                            
                            with col_chart2:
                                st.caption(f"15 Kata Paling Sering Muncul (**{fname}**)")
                                if not cache_vis['df_words'].empty:
                                    df_chart = cache_vis['df_words'].head(15)
                                    chart_words = alt.Chart(df_chart).mark_bar(color="#059669", cornerRadiusEnd=4).encode(
                                        y=alt.Y('Kata:N', sort='-x', title='Kata Kunci'),
                                        x=alt.X('Frekuensi:Q', title='Jumlah Muncul'),
                                        tooltip=[
                                            alt.Tooltip('Kata:N'), alt.Tooltip('Frekuensi:Q'),
                                            alt.Tooltip('Pasangan 1:N', title='🔗 Colloc 1'), 
                                            alt.Tooltip('Pasangan 2:N', title='🔗 Colloc 2'),
                                            alt.Tooltip('Pasangan 3:N', title='🔗 Collocation 3')
                                        ]
                                    ).properties(height=380, width=800).configure_view(stroke='#94A3B8', strokeWidth=1)
                                    st.altair_chart(chart_words, use_container_width=True)

                            with col_chart3:
                                st.caption(f"Word Cloud Dokumen (**{fname}**)")
                                if cache_vis['cloud_img']:
                                    st.image(cache_vis['cloud_img'], use_container_width=True)

                            # BARIS 2: Grafik N-Grams (Bigram & Trigram)
                            st.write("") # Spacer
                            col_ng1, col_ng2, col_ng_spacer = st.columns(3) # Menggunakan 3 kolom agar lebarnya seragam dgn yang atas
                            
                            with col_ng1:
                                st.caption(f"15 Frasa 2-Kata / Bigram (**{fname}**)")
                                if not cache_vis['df_bigram'].empty:
                                    ch_bi = alt.Chart(cache_vis['df_bigram']).mark_bar(color="#8B5CF6", cornerRadiusEnd=4).encode(
                                        y=alt.Y('Frasa:N', sort='-x', title='Bigram'),
                                        x=alt.X('Frekuensi:Q', title='Jumlah Muncul'),
                                        tooltip=['Frasa', 'Frekuensi']
                                    ).properties(height=380, width=800).configure_view(stroke='#94A3B8', strokeWidth=1)
                                    st.altair_chart(ch_bi, use_container_width=True)

                            with col_ng2:
                                st.caption(f"15 Frasa 3-Kata / Trigram (**{fname}**)")
                                if not cache_vis['df_trigram'].empty:
                                    ch_tri = alt.Chart(cache_vis['df_trigram']).mark_bar(color="#D946EF", cornerRadiusEnd=4).encode(
                                        y=alt.Y('Frasa:N', sort='-x', title='Trigram'),
                                        x=alt.X('Frekuensi:Q', title='Jumlah Muncul'),
                                        tooltip=['Frasa', 'Frekuensi']
                                    ).properties(height=380, width=800).configure_view(stroke='#94A3B8', strokeWidth=1)
                                    st.altair_chart(ch_tri, use_container_width=True)

                            # TABEL DATA 500 CONCORDANCE
                            if not cache_vis['df_words'].empty:
                                st.write("") 
                                with st.expander(f"🗃️ Tampilkan Tabel Data Concordance / Collocation (Top 500 Kata) - {fname}"):
                                    df_tabel_500 = cache_vis['df_words'].head(500)
                                    st.dataframe(df_tabel_500, use_container_width=True, height=300, hide_index=True)
                                    
                                    csv_500 = df_tabel_500.to_csv(index=False).encode('utf-8')
                                    st.download_button(label="📥 Download Data CSV (Top 500)", data=csv_500, file_name=f"concordance_{fname}.csv", mime="text/csv", key=f"dl_csv_{fname}")
                                    
                            st.markdown("<hr style='border: 1px dashed #E2E8F0; margin: 20px 0;'>", unsafe_allow_html=True)

            with tab_compare:
                st.markdown("<h3 style='color:#0F172A;'>⚖️ Perbandingan Antar Dokumen</h3>", unsafe_allow_html=True)
                st.caption("Bandingkan kosakata eksklusif antara dua dokumen untuk melihat perbedaan fokus topik.")
                
                if len(st.session_state.local_files) >= 2:
                    with st.container(border=True):
                        col_comp1, col_comp2 = st.columns(2)
                        opsi_semua_doc = list(st.session_state.local_files.keys())
                        
                        with col_comp1:
                            doc_a = st.selectbox("Pilih Dokumen A:", opsi_semua_doc, key="comp_doc_a")
                        with col_comp2:
                            def_idx = 1 if len(opsi_semua_doc) > 1 else 0
                            doc_b = st.selectbox("Pilih Dokumen B:", opsi_semua_doc, index=def_idx, key="comp_doc_b")
                        
                        if doc_a and doc_b:
                            if doc_a == doc_b:
                                st.warning("⚠️ Silakan pilih dua dokumen yang berbeda untuk dibandingkan.")
                            else:
                                with st.spinner("Menghitung matriks kemiripan..."):
                                    vocab_a = st.session_state.local_files[doc_a]['vocab']
                                    vocab_b = st.session_state.local_files[doc_b]['vocab']
                                    
                                    irisan = vocab_a.intersection(vocab_b)
                                    gabungan = vocab_a.union(vocab_b)
                                    jaccard_sim = len(irisan) / len(gabungan) if len(gabungan) > 0 else 0
                                    
                                    unik_a = vocab_a - vocab_b
                                    unik_b = vocab_b - vocab_a
                                    
                                    raw_a = re.findall(r'\b[a-z]+(?:-[a-z]+)*\b', st.session_state.local_files[doc_a]['cleaned'].lower())
                                    raw_b = re.findall(r'\b[a-z]+(?:-[a-z]+)*\b', st.session_state.local_files[doc_b]['cleaned'].lower())
                                    
                                    stop_words = nlp.Defaults.stop_words
                                    count_unik_a = Counter([w for w in raw_a if w in unik_a and w not in stop_words and len(w) > 2])
                                    count_unik_b = Counter([w for w in raw_b if w in unik_b and w not in stop_words and len(w) > 2])
                                    
                                    st.markdown("---")
                                    st.markdown(f"<h4 style='text-align:center; color:#334155; margin-bottom:5px;'>Tingkat Kemiripan Kosakata (Jaccard Index): <span style='color:#0EA5E9; font-size:24px;'>{jaccard_sim*100:.1f}%</span></h4>", unsafe_allow_html=True)
                                    st.progress(jaccard_sim)
                                    st.write("")
                                    
                                    c1, c2, c3 = st.columns(3)
                                    c1.metric(f"Total Kata Unik ({doc_a})", f"{len(vocab_a):,}")
                                    c2.metric("Kata Beririsan (Sama)", f"{len(irisan):,}")
                                    c3.metric(f"Total Kata Unik ({doc_b})", f"{len(vocab_b):,}")
                                    
                                    st.write("")
                                    col_res1, col_res2 = st.columns(2)
                                    
                                    def render_unique_words(doc_name, count_data, color_bg, color_border, icon):
                                        top_unik = count_data.most_common(30)
                                        pills_html = ""
                                        if top_unik:
                                            pills_html += "<div style='display:flex; gap:8px; flex-wrap:wrap; margin-top:20px;'>"
                                            for w, freq in top_unik:
                                                pills_html += f"<span style='background:white; color:#0F172A; border:1px solid {color_border}; border-radius:15px; padding:4px 12px; font-size:12.5px; font-weight:500; box-shadow: 0 1px 2px rgba(0,0,0,0.05);'>{w} <span style='color:#94A3B8; font-size:10.5px; margin-left:3px;'>({freq}x)</span></span>"
                                            pills_html += "</div>"
                                        else:
                                            pills_html = "<p style='font-style:italic; color:#64748B; margin-top:15px;'>Tidak ada kata eksklusif yang signifikan.</p>"

                                        full_html = f"""
                                        <div style='background:{color_bg}; padding:25px; border-radius:12px; border:1px solid {color_border}; height: 100%; box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.02);'>
                                            <h4 style='margin:0 0 8px 0; color:#0F172A; font-size:16.5px; font-weight:700;'>{icon} Eksklusif di Dokumen Ini</h4>
                                            <p style='margin:0; font-size:13.5px; color:#475569; line-height:1.5;'>
                                                Sering dibahas dalam <b>{doc_name}</b>, namun <u>tidak pernah</u> disebutkan di dokumen sebelahnya.
                                            </p>
                                            {pills_html}
                                        </div>
                                        """
                                        st.markdown(full_html, unsafe_allow_html=True)

                                    with col_res1:
                                        render_unique_words("Dokumen A", count_unik_a, "#F0F9FF", "#BAE6FD", "📘")
                                    with col_res2:
                                        render_unique_words("Dokumen B", count_unik_b, "#FDF4FF", "#FBCFE8", "📙")
                else:
                    st.info("ℹ️ Anda perlu mengunggah minimal 2 dokumen untuk menggunakan fitur Komparasi Dokumen.")
            with tab_search:
                st.markdown("<h3 style='color:#0F172A;'>🔍 Pencarian Pintar</h3>", unsafe_allow_html=True)
                with st.expander("ℹ️ Tentang Fitur & Cara Pakai"):
                    st.info("""
                        **Deskripsi:** Berbagai mode pencarian tingkat lanjut untuk membedah korpus Anda. Termasuk pencarian berdasarkan kelas kata (POS).
                        
                        **Fitur Dropdown Aksi (Di Setiap Kalimat):**
                        * **🏷️ POS Tag:** Membedah kalimat secara instan.
                        * **🌿 Syntax Tree:** Melihat gambar relasi antar kata (Dependency).
                        * **🌐 Trans:** Menerjemahkan kalimat terpilih saja.
                        """)

                col_mode, col_input, col_btn = st.columns([2.5, 4.5, 1], gap="small")
                
                with col_mode:
                    mode_pencarian = st.selectbox(
                        "Mode Pencarian", 
                        [
                            "🔍 Lemmatization", 
                            "🧠 Semantic Search", 
                            "🏷️ Entity Search (NER)", 
                            "📚 POS Search (Kelas Kata)", 
                            "🛒 Boolean Search", 
                            "⚙️ Regex Search", 
                            "🌳 Dependency Search"
                        ], 
                        label_visibility="collapsed"
                    )
                    
                with col_input:
                    if "NER" in mode_pencarian:
                        query_aktif = st.selectbox("Pilih Tipe Entitas", ["PERSON (Orang)", "ORG (Organisasi)", "GPE (Negara/Kota)", "LOC (Lokasi)", "DATE (Tanggal)", "MONEY (Keuangan)"], key="input_search_ner", label_visibility="collapsed")
                    
                    elif "POS Search" in mode_pencarian:
                        c_pos1, c_pos2 = st.columns([1, 1])
                        with c_pos1:
                            pos_label = st.selectbox("Pilih Kelas Kata:", list(MAP_SEMUA_POS.keys()), key="pos_tag_sel", label_visibility="collapsed")
                            target_pos_tag = MAP_SEMUA_POS[pos_label]
                        with c_pos2:
                            pos_keyword = st.text_input("Kata Kunci (Opsional):", placeholder="Ketik kata...", key="pos_kw_in", label_visibility="collapsed").strip().lower()
                        # Trigger unik untuk re-run
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
                        
                    elif "Semantic" in mode_pencarian:
                        query_aktif = st.text_input("Ketik konsep makna", placeholder="Ketik kata (misal: technology)...", label_visibility="collapsed").strip()
                    elif "Lemmatization" in mode_pencarian:
                        query_aktif = st.text_input("Ketik kata dasar", placeholder="Ketik kata dasar (misal: analyze)...", label_visibility="collapsed").strip()
                    elif "Boolean" in mode_pencarian:
                        query_aktif = st.text_input("Query Boolean", placeholder="Contoh: machine AND translation NOT neural", label_visibility="collapsed")

                with col_btn:
                    if "Regex" in mode_pencarian and query_aktif != "manual" and query_aktif != "":
                        st.markdown("<div style='margin-top: 2px;'></div>", unsafe_allow_html=True)
                    btn_cari = st.button("Cari", key="btn_search_key", use_container_width=True, type="primary")

                # State Inits
                if 'teks_pencarian' not in st.session_state: st.session_state.teks_pencarian = ""
                if 'search_results' not in st.session_state: st.session_state.search_results = []
                if 'current_page' not in st.session_state: st.session_state.current_page = 0
                if 'last_query' not in st.session_state: st.session_state.last_query = ""

                # --- PROSES PENCARIAN ---
                if query_aktif:
                    if st.session_state.last_query != query_aktif or btn_cari:
                        with st.spinner("Mencari & Memproses Struktur Kalimat (Harap Tunggu)..."):
                            matches_global = []
                            if "Lemmatization" in mode_pencarian or "Semantic" in mode_pencarian:
                                query_doc = nlp(query_aktif)
                            
                            for fname in selected_files:
                                teks_b = st.session_state.local_files[fname]['cleaned']
                                doc = nlp(teks_b[:100000])
                                
                                # Logika Khusus POS Search Baru
                                if "POS Search" in mode_pencarian:
                                    for s in doc.sents:
                                        match_found = False
                                        highlighted = ""
                                        for token in s:
                                            match_kw = (token.lemma_.lower() == pos_keyword) if pos_keyword else True
                                            match_pos = (token.pos_ == target_pos_tag) if target_pos_tag != "ALL" else True
                                            
                                            if match_kw and match_pos and (pos_keyword or target_pos_tag != "ALL"):
                                                match_found = True
                                                highlighted += f"<mark style='background:#0EA5E9; color:white; font-weight:bold; padding:0 4px; border-radius:4px;'>{token.text}</mark>{token.whitespace_}"
                                            else:
                                                highlighted += f"{token.text}{token.whitespace_}"
                                        
                                        if match_found:
                                            p_counts = Counter([t.pos_ for t in s if t.pos_ not in ['SPACE', 'PUNCT', 'SYM', 'X']])
                                            p_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>" + "".join([f"<span style='background-color: {Warna_POS_Utama.get(p, '#94A3B8')}; color:white; border-radius:4px; padding:3px 8px; font-size:11px; font-weight:600;'>{p}: {c}</span>" for p, c in p_counts.items()]) + "</div>"
                                            matches_global.append({'file': fname, 'text': s.text.strip(), 'html': highlighted, 'pills': p_html, 'tags': list(set([t.pos_ for t in s if t.pos_ in deskripsi_pos]))})
                                            
                                # Mode Lainnya Tetap Sama
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
                            st.session_state.current_page = 0
                            st.session_state.last_query = query_aktif

                # UI Display Hasil Search
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

            with tab_ngram:
                st.markdown("<h3 style='color:#0F172A;'>🔢 Analisis Frasa (N-Grams)</h3>", unsafe_allow_html=True)
                c_ng1, c_ng2 = st.columns([2, 5])
                with c_ng1:
                    t_ng = st.radio("Tipe N-Gram:", ["Bigram (2 Kata)", "Trigram (3 Kata)"])
                    n_val = 2 if "Bigram" in t_ng else 3
                    top_n = st.slider("Tampilkan Top:", 5, 30, 15)
                
                with c_ng2:
                    t_gabung_ng = " ".join([st.session_state.local_files[f]['cleaned'] for f in selected_files])
                    if t_gabung_ng:
                        with st.spinner(f"Mengekstrak {t_ng}..."):
                            w_ng = [w for w in re.findall(r'\b[a-z]{3,}\b', t_gabung_ng.lower()) if w not in nlp.Defaults.stop_words]
                            ng_list = [" ".join(g) for g in zip(*[w_ng[i:] for i in range(n_val)])]
                            c_ng = Counter(ng_list).most_common(top_n)
                            
                            if c_ng:
                                df_ng = pd.DataFrame(c_ng, columns=['Frasa', 'Frekuensi'])
                                ch_ng = alt.Chart(df_ng).mark_bar(color="#8B5CF6", cornerRadiusEnd=4).encode(y=alt.Y('Frasa:N', sort='-x'), x='Frekuensi:Q', tooltip=['Frasa', 'Frekuensi']).properties(height=400)
                                st.altair_chart(ch_ng, use_container_width=True)
                            else: st.warning("Kata tidak cukup.")
            # --- TAB 3: SUMMARIZATION ---
            with tab_summary:
                st.markdown("<h3 style='color:#0F172A;'>📝 Ekstraksi Dokumen Cepat (LexRank)</h3>", unsafe_allow_html=True)
                
                with st.expander("ℹ️ Tentang Fitur & Cara Pakai"):
                    st.info("""
                    **Deskripsi:** Menggunakan algoritma *LexRank* untuk mengekstrak kalimat-kalimat paling penting yang mewakili keseluruhan isi dokumen secara otomatis.
                    
                    **Cara Pakai:**
                    1. Pilih dokumen yang ingin diringkas dari *dropdown* di bawah.
                    2. Klik tombol **🚀 Mulai Ekstraksi Kilat**.
                    3. Gunakan menu titik tiga (**⋮**) di pojok kanan hasil untuk Salin atau Download.
                    """)
                st.caption("Karena proses peringkasan memakan banyak memori, fitur ini hanya berlaku untuk satu dokumen target yang Anda pilih di bawah ini.")
                
                # Memilih file mana yang mau diringkas
                target_sum_file = st.selectbox("Pilih dokumen spesifik untuk diringkas:", selected_files, key="sel_sum_file")
                
                if st.button(f"🚀 Mulai Ekstraksi Kilat", type="primary"):
                    with st.spinner(f"Mengekstrak informasi penting dari dokumen '{target_sum_file}'..."):
                        try:
                            teks_target = st.session_state.local_files[target_sum_file]['text']
                            tot_kalimat = st.session_state.local_files[target_sum_file]['stats']['k']
                            target_jumlah_kalimat = max(5, min(int(tot_kalimat * 0.15), 30))
                            
                            parser = PlaintextParser.from_string(teks_target, Tokenizer("english"))
                            summarizer_cepat = LexRankSummarizer()
                            hasil_ekstraksi = summarizer_cepat(parser.document, target_jumlah_kalimat)
                            
                            kalimat_terekstrak = [str(sentence) for sentence in hasil_ekstraksi]
                            
                            # Gabungkan kalimat menjadi paragraf untuk tampilan rapi
                            teks_plain_ringkasan = "\n\n".join(kalimat_terekstrak)
                            st.session_state.summary_results[target_sum_file] = teks_plain_ringkasan
                            
                            st.success(f"✅ Ekstraksi selesai!")
                        except Exception as e:
                            st.error(f"Terjadi kesalahan saat mengekstrak: {e}")

                # Tampilan Hasil
                if getattr(st.session_state, 'summary_results', None) and target_sum_file in st.session_state.summary_results:
                    teks_hasil = st.session_state.summary_results[target_sum_file]
                    
                    # Membungkus seluruh konten dalam satu container kartu putih
                    with st.container(border=True):
                        # Baris Header di dalam Kartu
                        col_judul, col_aksi = st.columns([0.9, 0.1])
                        
                        with col_judul:
                            st.markdown(f"#### 📑 Ringkasan: {target_sum_file}")
                            st.caption(f"📊 Estimasi: {len(teks_hasil.split())} kata | 📄 Dokumen: {target_sum_file}")

                        
                        with col_aksi:
                            with st.popover(""):
                                st.markdown("**Opsi Export**")
                
                                st.download_button(
                                    "📄 Download TXT", 
                                    data=teks_hasil.encode('utf-8'), 
                                    file_name=f"sum_{target_sum_file}.txt", 
                                    use_container_width=True
                                )
                                try:
                                    from docx import Document
                                    doc_ex = Document()
                                    doc_ex.add_heading(f"Ringkasan: {target_sum_file}", 0)
                                    
                                    paragraf_docx = teks_hasil.split('\n\n')
                                    for p in paragraf_docx:
                                        if p.strip():
                                            doc_ex.add_paragraph(p.strip())
                                    
                                    bio_docx = io.BytesIO()
                                    doc_ex.save(bio_docx)
                                    st.download_button(
                                        "📝 Download DOCX", 
                                        data=bio_docx.getvalue(), 
                                        file_name=f"sum_{target_sum_file}.docx", 
                                        use_container_width=True
                                    )
                                except: pass
                                
                        st.markdown("<hr style='margin: 10px 0; opacity: 0.2;'>", unsafe_allow_html=True)

                        st.markdown(f"""
                        <div style='color: #334155; font-size: 16px; line-height: 1.8; text-align: justify; padding: 10px 5px;'>
                            {teks_hasil.replace('\n\n', '<br><br>')}
                        </div>
                        """, unsafe_allow_html=True)
                  
                     
                    st.write("")
                    aksi_ringkasan = st.selectbox("Analisis Lanjutan:", ["Pilih Aksi...", "🏷️ POS Tagging", "🌐 Translate"], key="aksi_sum_last")
                    if aksi_ringkasan == "🏷️ POS Tagging":
                        with st.spinner("Membedah struktur kata..."):
                            st.markdown(get_colored_pos_text(teks_hasil), unsafe_allow_html=True)
                    
                    elif aksi_ringkasan == "🌐 Translate":
                        col_lang_sum, col_go_sum, _ = st.columns([2, 1, 5])
                        with col_lang_sum:
                            target_lang_sum_name = st.selectbox("Terjemahkan ke:", list(DAFTAR_BAHASA.keys()), key="lang_sum_utama", label_visibility="collapsed")
                        with col_go_sum:
                            go_trans_sum = st.button("Ok", key="go_sum_utama", use_container_width=True)
                            
                        if go_trans_sum:
                            target_lang_sum_code = DAFTAR_BAHASA[target_lang_sum_name]
                            with st.spinner(f"Menerjemahkan..."):
                                try:
                                    res_sum = GoogleTranslator(source='auto', target=target_lang_sum_code).translate(teks_hasil[:4500])
                                    st.markdown(f"""
                                    <div style='background-color: #F0FDF4; color: #065F46; padding: 25px; border-radius: 12px; font-size: 16px; line-height: 1.8; text-align: justify; border: 1px solid #A7F3D0; margin-top: 15px;'>
                                        {res_sum}
                                    </div>
                                    """, unsafe_allow_html=True)
                                except Exception as e: 
                                    st.error(f"Error: {e}")

            # --- TAB 4: WORD SKETCH ---
            with tab_sketch:
                st.markdown("<h3 style='color:#0F172A;'>🧩 Word Sketch (Profil Gramatikal Kata)</h3>", unsafe_allow_html=True)
                
                if "ws_input_key" not in st.session_state:
                    st.session_state["ws_input_key"] = ""

                def ganti_kata_pencarian(kata_baru):
                    st.session_state["ws_input_key"] = kata_baru

                col_ws_input, col_ws_btn = st.columns([4, 1.5], vertical_alignment="bottom")
                
                with col_ws_input:
                    st.text_input(
                        "Ketik kata yang ingin dibedah:", 
                        placeholder="Contoh: model, data, machine...",
                        key="ws_input_key" 
                    )
                
                with col_ws_btn:
                    st.button("Cari Profil Kata", type="primary", use_container_width=True)

                search_term = st.session_state["ws_input_key"]

                if search_term:
                    with st.spinner(f"Membangun profil untuk '{search_term}'..."):
                        teks_gabungan_ws = " ".join([st.session_state.local_files[f]['cleaned'] for f in selected_files])
                        hasil_ws = buat_word_sketch(search_term, teks_gabungan_ws)
                        
                        st.write(f"### 🔍 Hasil Word Sketch: **{search_term}**")
                        c_mod, c_subj, c_obj = st.columns(3)
                        
                        def render_clickable_list(header, data, color_hex, key_prefix):
                            st.markdown(f"<div style='background:{color_hex}; padding:10px; border-radius:8px; margin-bottom:10px;'><b>{header}</b></div>", unsafe_allow_html=True)
                            if data:
                                for i, (kata, jml) in enumerate(data):
                                    st.button(
                                        f"{kata} ({jml}x)", 
                                        key=f"{key_prefix}_{search_term}_{kata}_{i}", 
                                        use_container_width=True,
                                        on_click=ganti_kata_pencarian,
                                    )
                            else:
                                st.caption("Tidak ditemukan pola dominan.")

                        with c_mod:
                            render_clickable_list("✨ Modifiers", hasil_ws["✨ Modifiers (Sifat/Penjelas)"], "#F0FDF4", "mod")
                        with c_subj:
                            render_clickable_list("🏃‍♂️ Sbg Subjek", hasil_ws["🏃‍♂️ Sebagai Subjek (Melakukan)"], "#EFF6FF", "subj")
                        with c_obj:
                            render_clickable_list("🎯 Sbg Objek", hasil_ws["🎯 Sebagai Objek (Dikenai)"], "#FEF2F2", "obj")


        fitur_nlp_dashboard(selected_files)
else:
    st.info("👋 Silakan upload file terlebih dahulu untuk mulai menggunakan dashboard.")
