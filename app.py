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
                    # Filter kata yang terlalu pendek atau berupa kata sambung (stop words)
                    if len(kandidat) > 2 and kandidat not in stop_words:
                        pasangan.append(kandidat)
                        
    # Mengembalikan 5 kata yang paling sering berdampingan
    return Counter(pasangan).most_common(5)

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

uploaded_files = st.file_uploader(
    "Upload File (PDF, DOCX, TXT)", 
    accept_multiple_files=True, 
    type=['pdf', 'docx', 'txt'],
    key="file_uploader_widget"
)

if st.session_state.local_files:
    current_uploader_filenames = [f.name for f in uploaded_files] if uploaded_files else []
    
    files_to_remove = [f for f in st.session_state.local_files if f not in current_uploader_filenames]
    
    for f in files_to_remove:
        del st.session_state.local_files[f]
        if f in st.session_state.summary_results:
            del st.session_state.summary_results[f]
    
    if files_to_remove:
        st.rerun()

if uploaded_files:
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

if st.session_state.local_files:
    file_names = list(st.session_state.local_files.keys())
    
    col_sel1, col_sel2 = st.columns([8, 2], vertical_alignment="bottom")
    
    with col_sel1:
        st.markdown("<div style='font-size:14px; font-weight:600; color:#475569; margin-bottom:8px;'>Pilih Dokumen Aktif:</div>", unsafe_allow_html=True)
        
        # Logika default jika tombol "Pilih Semua" dicentang
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
        # 4. VISUALISASI UTAMA
        # ==========================================
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

        # KUNCI 1: Cache gambar WordCloud agar tidak digambar ulang (bikin lemot) tiap klik Next
        @st.cache_data(show_spinner=False)
        def get_cached_wordcloud(text_data):
            wc = WordCloud(width=600, height=280, background_color='white', colormap='viridis', max_words=300).generate(text_data)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            return fig

        with st.expander("Klik di sini untuk Membuka / Menutup Visualisasi Data", expanded=False):
            with st.container():
                for fname in selected_files:
                    st.markdown(f"#### 📄 Laporan: {fname}")
                    teks_dokumen = st.session_state.local_files[fname]['text']
                    df_pos, df_words, df_cloud = dapatkan_data_visual(teks_dokumen[:80000])
                    
                    col_chart1, col_chart2, col_chart3 = st.columns(3)
                
                    with col_chart1:
                        st.caption(f"Statistik Tata Bahasa (**{fname}**)")
                        if not df_pos.empty:
                            chart_pos = alt.Chart(df_pos).mark_bar(color="#0284C7", cornerRadiusEnd=4).encode(
                                y=alt.Y('POS Tag', sort='-x', title='Kelas Kata'), 
                                x=alt.X('Jumlah Kata', title='Total Jumlah'),
                                tooltip=['POS Tag', 'Jumlah Kata']
                            ).properties(height=380, width=800).configure_view(stroke='#94A3B8', strokeWidth=1)
                            st.altair_chart(chart_pos, use_container_width=True)
                    
                    with col_chart2:
                        st.caption(f"15 Kata Paling Sering Muncul (**{fname}**)")
                        if not df_words.empty:
                            teks_mentah_aktif = st.session_state.local_files[fname]['cleaned']
                            
                            if 'Pasangan 1' not in df_words.columns:
                                col1, col2, col3, col4, col5 = [], [], [], [], []
                                for i, kata in enumerate(df_words['Kata']):
                                    if i < 8000: 
                                        hasil_colloc = hitung_collocation(kata, teks_mentah_aktif, window=5)
                                        col1.append(f"{hasil_colloc[0][0]} ({hasil_colloc[0][1]}x)" if len(hasil_colloc) > 0 else "-")
                                        col2.append(f"{hasil_colloc[1][0]} ({hasil_colloc[1][1]}x)" if len(hasil_colloc) > 1 else "-")
                                        col3.append(f"{hasil_colloc[2][0]} ({hasil_colloc[2][1]}x)" if len(hasil_colloc) > 2 else "-")
                                        col4.append(f"{hasil_colloc[3][0]} ({hasil_colloc[3][1]}x)" if len(hasil_colloc) > 3 else "-")
                                        col5.append(f"{hasil_colloc[4][0]} ({hasil_colloc[4][1]}x)" if len(hasil_colloc) > 4 else "-")
                                    else:
                                        col1.append("-"); col2.append("-"); col3.append("-"); col4.append("-"); col5.append("-")
                                df_words['Pasangan 1'] = col1; df_words['Pasangan 2'] = col2; df_words['Pasangan 3'] = col3; df_words['Pasangan 4'] = col4; df_words['Pasangan 5'] = col5

                            unik_klik_nama = f"KlikBar_{fname.replace('.','')}"
                            klik_bar = alt.selection_point(fields=['Kata'], empty=False, name=unik_klik_nama)

                            chart_words = alt.Chart(df_words).transform_window(
                                rank='row_number()', sort=[alt.SortField("Frekuensi", order="descending")]
                            ).transform_filter(alt.datum.rank <= 15).mark_bar(color="#059669", cornerRadiusEnd=4).encode(
                                y=alt.Y('Kata:N', sort='-x', title='Kata Kunci'),
                                x=alt.X('Frekuensi:Q', title='Jumlah Muncul'),
                                tooltip=[
                                    alt.Tooltip('Kata:N'), alt.Tooltip('Frekuensi:Q'),
                                    alt.Tooltip('Pasangan 1:N', title='🔗 Collocation 1'), 
                                    alt.Tooltip('Pasangan 2:N', title='🔗 Collocation 2'),
                                    alt.Tooltip('Pasangan 3:N', title='🔗 Collocation 3'),
                                    alt.Tooltip('Pasangan 4:N', title='🔗 Collocation 4'),
                                    alt.Tooltip('Pasangan 5:N', title='🔗 Collocation 5')
                                ]
                            ).properties(height=380, width=800).configure_view(stroke='#94A3B8', strokeWidth=1)
                            
                            st.altair_chart(chart_words, use_container_width=True)

                    with col_chart3:
                        st.caption(f"Word Cloud Dokumen (**{fname}**)")
                        if df_cloud:
                            # Panggil dari cache agar instan!
                            fig_cloud = get_cached_wordcloud(df_cloud[:80000])
                            st.pyplot(fig_cloud, use_container_width=True)
                            plt.close(fig_cloud)
        st.write("")
        st.markdown("<br>", unsafe_allow_html=True)
                
        # ==========================================
        # 5. TAB FITUR NLP 
        # ==========================================
        @st.fragment
        def fitur_nlp_dashboard(selected_files):

            def prev_page_1(): st.session_state.current_page -= 1
            def next_page_1(): st.session_state.current_page += 1
            def prev_page_2(): st.session_state.ps_current_page -= 1
            def next_page_2(): st.session_state.ps_current_page += 1

            tab_search, tab_pos_search, tab_summary = st.tabs([
                "🔍 Search Words", "🕵️‍♂️ Search Grammar", "📝 Summarization"
            ])
            def render_result_cards(results, query_lemma, current_page, items_per_page, is_pos_search=False, target_tag=None):
                start = current_page * items_per_page
                end = start + items_per_page
                subset = results[start:end]

                for i, match in enumerate(subset):
                    col_c, col_a = st.columns([8.5, 1.5], gap="small")
                    with col_a:
                        aksi = st.selectbox("Aksi", ["Aksi", "🏷️ POS Tag", "🌐 Trans"], key=f"ak_{i}_{start}", label_visibility="collapsed")
                    
                    with col_c:
                        doc_m = nlp(match)
                        counts = Counter([t.pos_ for t in doc_m if t.pos_ not in ['SPACE', 'PUNCT', 'SYM', 'X']])
                        
                        highlighted = ""
                        for token in doc_m:
                            match_logic = (token.lemma_.lower() == query_lemma) if query_lemma else True
                            if is_pos_search: match_logic = match_logic and (token.pos_ == target_tag or target_tag == "ALL")
                            
                            if match_logic and query_lemma:
                                color = "#EF4444" if is_pos_search else "#FDE047"
                                text_color = "white" if is_pos_search else "#0F172A"
                                highlighted += f"<mark style='background:{color}; color:{text_color}; font-weight:bold; padding:0 4px; border-radius:3px;'>{token.text}</mark>{token.whitespace_}"
                            else:
                                highlighted += f"{token.text}{token.whitespace_}"

                        pills_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>"
                        for pos in sorted(counts.keys()):
                            bg = Warna_POS_Utama.get(pos, '#94A3B8')
                            pills_html += f"<span style='background-color: {bg}; color:white; border-radius:4px; padding:2px 8px; font-size:11px; font-weight:600;'>{pos}: {counts[pos]}</span>"
                        pills_html += "</div>"

                        st.markdown(f"""<div style='background:#FFFFFF; padding:20px; border-radius:10px; border:1px solid #E2E8F0; margin-bottom:10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                            <div style='color:#334155; font-size:15px; margin-bottom:20px; line-height:1.7;'>{highlighted}</div>
                            <div style='border-top:1px solid #F1F5F9; padding-top:12px; display:flex; justify-content:space-between; align-items:center;'>
                                <div style='font-size:11px; color:#0284C7; font-weight:bold; background:#F0F9FF; padding:3px 8px; border-radius:4px;'>📄 DOKUMEN: {matches_global.upper()}</div>
                                <div>{pills_html}</div>
                            </div></div>""", unsafe_allow_html=True)
                        
                        if aksi == "🏷️ POS Tag":
                            st.markdown(get_colored_pos_text(match), unsafe_allow_html=True); st.write("")
                            st.write("---")
                            info_tag = st.selectbox(
                                "💡 Bingung dengan label di atas? Pilih untuk penjelasan:", 
                                options=list(deskripsi_pos.keys()),
                                key=f"help_{i}_{start}"
                            )
                            st.info(deskripsi_pos[info_tag])
                            st.write("")

                        elif aksi == "🌐 Trans":
                            l1, l2, _ = st.columns([3, 1, 6])
                            with l1: target_lang = st.selectbox("Ke:", list(DAFTAR_BAHASA.keys()), key=f"tr_{i}_{start}", label_visibility="collapsed")
                            with l2: go = st.button("Ok", key=f"go_{i}_{start}", use_container_width=True)
                            if go:
                                res = GoogleTranslator(source='auto', target=DAFTAR_BAHASA[target_lang]).translate(match)
                                st.markdown(f"<div style='background:#E0F2FE; border: 1px solid #7DD3FC; padding:15px; color:#0369A1; border-radius:8px; font-size:15px; margin-bottom:15px;'>{res}</div>", unsafe_allow_html=True)
                                
            # --- TAB 1: SEARCHING ---
            # --- TAB 1: SEARCHING ---
            with tab_search:
                st.markdown("<h3 style='color:#0F172A;'>🔍 Pencarian Pintar</h3>", unsafe_allow_html=True)

                with st.expander("ℹ️ Tentang Fitur & Cara Pakai"):
                    st.info("""
                        **Deskripsi:** Mencari kata berdasarkan *Lemma* (kata dasar) atau *Semantic Search* (kemiripan makna).
                        
                        **Fitur Dropdown Aksi (Di Setiap Kalimat):**
                        * **🏷️ POS Tag:** Membedah kalimat secara instan.
                        * **🌐 Trans:** Menerjemahkan kalimat terpilih saja.
                        """)

                col_input, col_mode, col_btn = st.columns([5.5, 1.5, 1], gap="small")
                
                with col_input:
                    query_aktif = st.text_input(
                        "Pencarian", 
                        key="input_search_key",
                        placeholder="Ketik kata dan tekan Enter...", 
                        label_visibility="collapsed"
                    )
                with col_mode:
                    mode_pencarian = st.selectbox(
                        "Mode Pencarian", 
                        ["🔍 Lemmatization (Kata Dasar)", "🧠 Semantic Search (Makna)"], 
                        label_visibility="collapsed"
                    )
                with col_btn:
                    btn_cari = st.button("Cari", key="btn_search_key", use_container_width=True, type="primary")

                if 'teks_pencarian' not in st.session_state: st.session_state.teks_pencarian = ""
                if 'query_lemma' not in st.session_state: st.session_state.query_lemma = "" 
                if 'search_results' not in st.session_state: st.session_state.search_results = []
                if 'current_page' not in st.session_state: st.session_state.current_page = 0
                if 'last_query' not in st.session_state: st.session_state.last_query = ""

                if query_aktif:
                    if st.session_state.last_query != query_aktif or btn_cari:
                        with st.spinner("Mencari & Memproses Struktur Kalimat (Harap Tunggu)..."):
                            query_doc = nlp(query_aktif.strip())
                            matches_global = []
                            
                            for fname in selected_files:
                                teks_b = st.session_state.local_files[fname]['cleaned']
                                doc = nlp(teks_b[:100000])
                                
                                if "Lemmatization" in mode_pencarian:
                                    query_lemma = query_doc[0].lemma_.lower() if len(query_doc) > 0 else query_aktif.lower()
                                    st.session_state.query_lemma = query_lemma
                                    
                                    for s in doc.sents:
                                        if any(token.lemma_.lower() == query_lemma for token in s):
                                            match_text = s.text.strip()
                                            
                                            # -- PRE-COMPUTE: Highlight --
                                            kata_cocok = set([t.text for t in s if t.lemma_.lower() == query_lemma])
                                            if kata_cocok:
                                                pola_regex = r"\b(" + "|".join(map(re.escape, kata_cocok)) + r")\b"
                                                highlighted = re.sub(pola_regex, r"<mark style='background:#0EA5E9; color:white; font-weight:bold; padding:0 4px; border-radius:3px;'>\1</mark>", match_text, flags=re.I)
                                            else:
                                                highlighted = match_text
                                                
                                            # -- PRE-COMPUTE: Pills & Tags --
                                            pos_counts = Counter([token.pos_ for token in s if token.pos_ not in ['SPACE', 'PUNCT', 'SYM', 'X']])
                                            pills_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>"
                                            for pos, count in pos_counts.items():
                                                bg_color = Warna_POS_Utama.get(pos, '#94A3B8')
                                                pills_html += f"<span style='background-color: {bg_color}; color:white; border-radius:4px; padding:3px 8px; font-size:11px; font-weight:600;'>{pos.capitalize()}: {count}</span>"
                                            pills_html += "</div>"
                                            
                                            tags_di_kalimat = list(set([token.pos_ for token in s if token.pos_ in deskripsi_pos]))
                                            
                                            matches_global.append({
                                                'file': fname, 
                                                'text': match_text, 
                                                'html': highlighted, 
                                                'pills': pills_html,
                                                'tags': tags_di_kalimat
                                            })
                                else:
                                    st.session_state.query_lemma = "" 
                                    for s in doc.sents:
                                        if len(s.text.strip()) > 5:
                                            is_similar = False
                                            for token in s:
                                                if not token.is_stop and not token.is_punct and token.has_vector:
                                                    if query_doc.similarity(token) >= 0.60: 
                                                        is_similar = True
                                                        break 
                                            if is_similar:
                                                match_text = s.text.strip()
                                                pos_counts = Counter([token.pos_ for token in s if token.pos_ not in ['SPACE', 'PUNCT', 'SYM', 'X']])
                                                pills_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>"
                                                for pos, count in pos_counts.items():
                                                    bg_color = Warna_POS_Utama.get(pos, '#94A3B8')
                                                    pills_html += f"<span style='background-color: {bg_color}; color:white; border-radius:4px; padding:3px 8px; font-size:11px; font-weight:600;'>{pos.capitalize()}: {count}</span>"
                                                pills_html += "</div>"
                                                tags_di_kalimat = list(set([token.pos_ for token in s if token.pos_ in deskripsi_pos]))

                                                matches_global.append({
                                                    'file': fname, 
                                                    'text': match_text, 
                                                    'html': match_text, 
                                                    'pills': pills_html,
                                                    'tags': tags_di_kalimat
                                                })

                            st.session_state.search_results = matches_global
                            st.session_state.current_page = 0
                            st.session_state.last_query = query_aktif

                    if not st.session_state.search_results:
                        st.warning(f"🔍 Kata '{query_aktif}' tidak ditemukan dalam isi dokumen.")

                if query_aktif and "Lemmatization" in mode_pencarian:
                    sinonim_set = set()
                    try:
                        for syn in wordnet.synsets(query_aktif):
                            for l in syn.lemmas():
                                kata_terkait = l.name().lower()
                                if kata_terkait != query_aktif.lower() and '_' not in kata_terkait:
                                    sinonim_set.add(kata_terkait)
                    except: pass
                    
                    vocab_gabungan = set()
                    for fname in selected_files: 
                        vocab_gabungan.update(st.session_state.local_files[fname]['vocab'])
                    
                    saran_kata = [kata for kata in sinonim_set if kata in vocab_gabungan][:8]
                    if saran_kata:
                        st.markdown("<div style='font-size:14px; color:#64748B; margin-bottom:8px;'>💡 Saran kata terkait yang <b>ada di dokumen ini</b>:</div>", unsafe_allow_html=True)
                        kunci_dinamis_pills = f"saran_pills_{query_aktif}_{len(selected_files)}"
                        
                        def aksi_klik_saran_dinamis():
                            pilihan = st.session_state.get(kunci_dinamis_pills)
                            if pilihan:
                                st.session_state.input_search_key = pilihan
                                st.session_state.current_page = 0
                                
                        try:
                            st.pills("Saran", saran_kata, key=kunci_dinamis_pills, on_change=aksi_klik_saran_dinamis, label_visibility="collapsed")
                        except AttributeError:
                            pass

                if st.session_state.search_results:
                    total_results = len(st.session_state.search_results)
                    
                    col_info, col_blank, col_filter_text, col_filter_drop = st.columns([5, 1, 3, 1.05], gap="small")
                    with col_info:
                        st.markdown(f"<div style='color:#059669; font-weight:bold; font-size:14.5px; padding-top:8px;'>✅ Ditemukan {total_results} kalimat.</div>", unsafe_allow_html=True)
                    with col_filter_text:
                        st.markdown("<div style='text-align:right; padding-top:8px; font-size:14px; color:#475569; font-weight:00;'>Tampilkan per halaman:</div>", unsafe_allow_html=True)
                    with col_filter_drop:
                        opsi_dasar = [5, 10, 25, 50, 100]
                        opsi_limit = [str(x) for x in opsi_dasar if x < total_results]
                        opsi_limit.append(f"All ({total_results})")
                        pilihan_limit = st.selectbox("Limit", opsi_limit, key="limit_search_t1", label_visibility="collapsed")
                    
                    if pilihan_limit.startswith("All"): ITEMS_PER_PAGE = total_results if total_results > 0 else 1
                    else: ITEMS_PER_PAGE = int(pilihan_limit)
                        
                    total_pages = max(1, math.ceil(total_results / ITEMS_PER_PAGE))
                    if st.session_state.current_page >= total_pages: st.session_state.current_page = max(0, total_pages - 1)
                        
                    start_idx = st.session_state.current_page * ITEMS_PER_PAGE
                    end_idx = start_idx + ITEMS_PER_PAGE
                    subset_results = st.session_state.search_results[start_idx:end_idx]

                    st.write("") 

                    # ==========================================
                    # LOOPING RENDER KARTU HASIL (CEPAT)
                    # ==========================================
                    for i, match_data in enumerate(subset_results):
                        with st.container(border=True):
                            st.markdown(f"<div style='color:#334155; font-size:15.5px; margin-bottom:15px; line-height:1.6;'>{match_data['html']}</div>", unsafe_allow_html=True)
                            
                            col_kiri, col_spacer, col_kanan = st.columns([7, 1, 1.18], vertical_alignment="center")                        
                            with col_kiri:
                                gabungan_html = f"""
                                <div style='display:flex; align-items:center; gap:12px; flex-wrap:wrap;'>
                                    <div style='font-size:11px; color:#0284C7; font-weight:bold; background:#F0F9FF; padding:5px 10px; border-radius:4px; border:1px solid #BAE6FD; white-space:nowrap;'>
                                        📄 {match_data['file'].upper()}
                                    </div>
                                    <div>
                                        {match_data['pills']}
                                    </div>
                                </div>
                                """
                                st.markdown(gabungan_html, unsafe_allow_html=True)
                            
                            with col_spacer:
                                st.write("")
                            
                            with col_kanan:
                                aksi = st.selectbox("Aksi", ["Aksi", "🏷️ POS Tag", "🌐 Trans"], key=f"aksi_t1_{start_idx + i}", label_visibility="collapsed")

                            if aksi == "🏷️ POS Tag":
                                st.markdown(get_colored_pos_text(match_data['text']), unsafe_allow_html=True); st.write("")
                                opsi_dropdown = [tag for tag in deskripsi_pos.keys() if tag in match_data['tags']]
                                if opsi_dropdown:
                                    info_tag = st.selectbox("💡 Penjelasan label pada kalimat ini:", options=opsi_dropdown, key=f"help_t1_{start_idx + i}")
                                    st.info(deskripsi_pos[info_tag])
                                st.write("")
                            
                            elif aksi == "🌐 Trans":
                                col_lang, col_go = st.columns([7, 3])
                                with col_lang:
                                    target_lang_name = st.selectbox("Ke:", list(DAFTAR_BAHASA.keys()), key=f"lang_t1_{start_idx + i}", label_visibility="collapsed")
                                with col_go:
                                    go_trans = st.button("Ok", key=f"go_t1_{start_idx + i}", use_container_width=True)
                                if go_trans:
                                    target_lang_code = DAFTAR_BAHASA[target_lang_name]
                                    with st.spinner("Menerjemahkan..."):
                                        try:
                                            hasil_terjemahan = GoogleTranslator(source='auto', target=target_lang_code).translate(match_data['text'])
                                            st.markdown(f"<div style='background:#E0F2FE; border: 1px solid #7DD3FC; padding:15px; color:#0369A1; border-radius:8px; font-size:15px; margin-top:10px; font-weight:500;'>{hasil_terjemahan}</div>", unsafe_allow_html=True)
                                        except Exception as e: 
                                            st.error(f"Error: {e}")

                    # ==========================================
                    # KONTROL PAGINATION (SUDAH DI LUAR LOOP)
                    # ==========================================
                    st.write("") 
                    col_blank1, col_p, col_page_info, col_n, col_blank2 = st.columns([2, 1, 2, 1, 2])
                    with col_p:
                        # MENGGUNAKAN CALLBACK (ON_CLICK) AGAR TIDAK PERLU ST.RERUN()
                        st.button("⬅️ Prev", key="prev_btn_1", use_container_width=True, disabled=(st.session_state.current_page == 0), on_click=prev_page_1)
                    with col_page_info:
                        st.markdown(f"<div style='text-align:center; padding-top:8px; font-size:14px; color:#475569;'>Halaman: <b>{st.session_state.current_page + 1} / {total_pages}</b></div>", unsafe_allow_html=True)
                    with col_n:
                        # MENGGUNAKAN CALLBACK (ON_CLICK) AGAR TIDAK PERLU ST.RERUN()
                        st.button("Next ➡️", key="next_btn_1", use_container_width=True, disabled=(st.session_state.current_page >= total_pages - 1), on_click=next_page_1)

            with tab_pos_search:
                st.markdown("<h3 style='color:#0F172A;'>🕵️‍♂️ Pencarian Spesifik (POS Search)</h3>", unsafe_allow_html=True)

                with st.expander("ℹ️ Tentang Fitur & Cara Pakai"):
                    st.info("""
                    **Deskripsi:** Mencari kalimat berdasarkan peran tata bahasa (*Part-of-Speech*).
                    1. Pilih **Kelas Kata** yang ingin dicari.
                    2. Masukkan **Kata Kunci** jika ingin mencari kata spesifik.
                    """)

                st.caption("Bisa diisi salah satu (hanya kelas kata, atau hanya kata kunci), atau isi dua-duanya.")
                
                if 'ps_query' not in st.session_state: st.session_state.ps_query = ""
                if 'ps_tag' not in st.session_state: st.session_state.ps_tag = "ALL"
                if 'ps_results' not in st.session_state: st.session_state.ps_results = []
                if 'ps_current_page' not in st.session_state: st.session_state.ps_current_page = 0
                
                col_ps1, col_ps2, col_ps3 = st.columns([4, 4, 1.4], gap="small")
                with col_ps1:
                    ps_label_pilihan = st.selectbox("Pilih Kelas Kata:", list(MAP_SEMUA_POS.keys()), key="input_ps_tag")
                    ps_target_tag = MAP_SEMUA_POS[ps_label_pilihan]
                with col_ps2:
                    ps_keyword = st.text_input("Kata Kunci (Opsional):", placeholder="Ketik kata (misal: Learn)...", key="input_ps_keyword")
                with col_ps3:
                    st.markdown("<div style='margin-top: 27.5px;'></div>", unsafe_allow_html=True)
                    btn_ps_cari = st.button("Cari Presisi", key="btn_cari_t2", type="primary", use_container_width=True)

                if ps_keyword or ps_target_tag != "ALL":
                    current_key = f"{ps_target_tag}_{ps_keyword}_{len(selected_files)}"
                    
                    if btn_ps_cari or st.session_state.get('last_ps_key') != current_key:            
                        st.session_state.ps_query = ps_keyword.strip()
                        st.session_state.ps_tag = ps_target_tag
                        
                        with st.spinner("Mencari & Memproses Struktur Kalimat (Harap Tunggu)..."):
                            q_lemma = nlp(st.session_state.ps_query)[0].lemma_.lower() if st.session_state.ps_query else ""
                            ps_matches_global = []
                            
                            for fname in selected_files:
                                doc = nlp(st.session_state.local_files[fname]['cleaned'][:100000])
                                for s in doc.sents:
                                    match_found = False
                                    highlighted_html = ""
                                    pos_counts = Counter()
                                    
                                    for token in s:
                                        if token.pos_ not in ['SPACE', 'PUNCT', 'SYM', 'X']:
                                            pos_counts[token.pos_] += 1
                                        
                                        match_kw = (token.lemma_.lower() == q_lemma) if q_lemma else True
                                        match_pos = (token.pos_ == st.session_state.ps_tag) if st.session_state.ps_tag != "ALL" else True
                                        
                                        if match_kw and match_pos and (q_lemma or st.session_state.ps_tag != "ALL"):
                                            match_found = True
                                            highlighted_html += f"<mark style='background:#0EA5E9; color:white; font-weight:bold; padding:0 4px; border-radius:4px; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>{token.text}</mark>{token.whitespace_}"
                                        else:
                                            highlighted_html += f"{token.text}{token.whitespace_}"
                                            
                                    if match_found:
                                        # -- PRE-COMPUTE PILLS --
                                        pills_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>"
                                        for pos, count in pos_counts.most_common():
                                            bg_color = Warna_POS_Utama.get(pos, '#94A3B8')
                                            pills_html += f"<span style='background-color: {bg_color}; color:white; border-radius:4px; padding:3px 8px; font-size:11px; font-weight:600;'>{pos.capitalize()}: {count}</span>"
                                        pills_html += "</div>"
                                        
                                        tags_di_kalimat = list(set([token.pos_ for token in s if token.pos_ in deskripsi_pos]))

                                        ps_matches_global.append({
                                            'file': fname, 
                                            'text': s.text.strip(),
                                            'html': highlighted_html,
                                            'pills': pills_html,
                                            'tags': tags_di_kalimat
                                        })
                            
                            st.session_state.ps_results = ps_matches_global
                            st.session_state.ps_current_page = 0
                            st.session_state.last_ps_key = current_key

                if (ps_keyword or ps_target_tag != "ALL") and not st.session_state.ps_results:
                    st.warning(f"🕵️‍♂️ Tidak ditemukan kata yang sesuai di isi utama dokumen.")

                if st.session_state.ps_results:
                    ps_total = len(st.session_state.ps_results)
                    
                    if st.session_state.ps_query and st.session_state.ps_tag != "ALL": info_msg = f"✅ Ditemukan {ps_total} kalimat di mana '{st.session_state.ps_query}' berperan sebagai **{st.session_state.ps_tag}**."
                    elif st.session_state.ps_query: info_msg = f"✅ Ditemukan {ps_total} kalimat yang mengandung kata '{st.session_state.ps_query}'."
                    else: info_msg = f"✅ Ditemukan {ps_total} kalimat yang memiliki struktur **{st.session_state.ps_tag}**."
                        
                    col_info_ps, col_blank_ps, col_filter_text_ps, col_filter_drop_ps = st.columns([6, 0.5, 3, 1.05], gap="small")
                    with col_info_ps: st.markdown(f"<div style='color:#059669; font-weight:bold; font-size:14.5px; padding-top:8px;'>{info_msg}</div>", unsafe_allow_html=True)
                    with col_filter_text_ps: st.markdown("<div style='text-align:right; padding-top:8px; font-size:14px; color:#475569; font-weight:00;'>Tampilkan per halaman:</div>", unsafe_allow_html=True)
                    with col_filter_drop_ps:
                        ps_opsi_dasar = [5, 10, 25, 50, 100]
                        ps_opsi_limit = [str(x) for x in ps_opsi_dasar if x < ps_total]
                        ps_opsi_limit.append(f"All ({ps_total})")
                        ps_pilihan_limit = st.selectbox("Limit PS", ps_opsi_limit, key="limit_ps_search_t2", label_visibility="collapsed")
                    
                    if ps_pilihan_limit.startswith("All"): PS_ITEMS_PER_PAGE = ps_total if ps_total > 0 else 1
                    else: PS_ITEMS_PER_PAGE = int(ps_pilihan_limit)
                        
                    ps_total_pages = max(1, math.ceil(ps_total / PS_ITEMS_PER_PAGE))
                    if st.session_state.ps_current_page >= ps_total_pages: st.session_state.ps_current_page = max(0, ps_total_pages - 1)
                        
                    ps_start = st.session_state.ps_current_page * PS_ITEMS_PER_PAGE
                    ps_end = ps_start + PS_ITEMS_PER_PAGE
                    ps_subset = st.session_state.ps_results[ps_start:ps_end]
                    
                    st.write("") 

                    # ==========================================
                    # LOOPING RENDER KARTU HASIL TAB 2
                    # ==========================================
                    for j, match_data in enumerate(ps_subset):
                        with st.container(border=True):
                            st.markdown(f"<div style='color:#334155; font-size:15.5px; margin-bottom:15px; line-height:1.6;'>{match_data['html']}</div>", unsafe_allow_html=True)
                            
                            tanda_pos = st.session_state.ps_tag if st.session_state.ps_tag != "ALL" else "BEBAS"
                            col_kiri_ps, col_spacer_ps, col_kanan_ps = st.columns([7, 1, 1.18], vertical_alignment="center")
                            
                            with col_kiri_ps:
                                gabungan_html_ps = f"""
                                <div style='display:flex; align-items:center; gap:12px; flex-wrap:wrap;'>
                                    <div style='font-size:11px; color:white; font-weight:bold; background:#0EA5E9; padding:5px 10px; border-radius:4px; border:1px solid #FECACA; white-space:nowrap;'>
                                        🎯 TARGET: {tanda_pos}
                                    </div>
                                    <div>
                                        {match_data['pills']}
                                    </div>
                                </div>
                                """
                                st.markdown(gabungan_html_ps, unsafe_allow_html=True)
                                
                            with col_spacer_ps:
                                st.write("")
                                
                            with col_kanan_ps:
                                ps_aksi = st.selectbox("Aksi", ["Aksi", "🏷️ POS Tag", "🌐 Trans"], key=f"ps_aksi_t2_{ps_start + j}", label_visibility="collapsed")
                            
                            if ps_aksi == "🏷️ POS Tag":
                                st.markdown(get_colored_pos_text(match_data['text']), unsafe_allow_html=True)
                                opsi_dropdown_ps = [tag for tag in deskripsi_pos.keys() if tag in match_data['tags']]
                                if opsi_dropdown_ps:
                                    ps_info_tag = st.selectbox("💡 Penjelasan label pada kalimat ini:", options=opsi_dropdown_ps, key=f"help_t2_{ps_start + j}")
                                    st.info(deskripsi_pos[ps_info_tag])
                                st.write("")

                            elif ps_aksi == "🌐 Trans":
                                col_lang, col_go, _ = st.columns([4, 2, 4])
                                with col_lang:
                                    t_lang_name = st.selectbox("Ke:", list(DAFTAR_BAHASA.keys()), key=f"ps_lang_t2_{ps_start + j}", label_visibility="collapsed")
                                with col_go:
                                    ps_go_trans = st.button("Ok", key=f"ps_go_t2_{ps_start + j}", use_container_width=True)
                                if ps_go_trans:
                                    t_lang_code = DAFTAR_BAHASA[t_lang_name]
                                    with st.spinner("Menerjemahkan..."):
                                        try:
                                            h_trans = GoogleTranslator(source='auto', target=t_lang_code).translate(match_data['text'])
                                            st.markdown(f"<div style='background:#E0F2FE; border: 1px solid #7DD3FC; padding:15px; color:#0369A1; border-radius:8px; font-size:15px; margin-top:10px; font-weight:500;'>{h_trans}</div>", unsafe_allow_html=True)
                                        except Exception as e: st.error(f"Error: {e}")

                    # ==========================================
                    # KONTROL PAGINATION (SUDAH DI LUAR LOOP)
                    # ==========================================
                    st.write("") 
                    col_b1, col_pp, col_pi, col_pn, col_b2 = st.columns([2, 1, 2, 1, 2])
                    with col_pp:
                        # MENGGUNAKAN CALLBACK (ON_CLICK) AGAR TIDAK PERLU ST.RERUN()
                        st.button("⬅️ Prev", key="ps_prev_btn_2", use_container_width=True, disabled=(st.session_state.ps_current_page == 0), on_click=prev_page_2)
                    with col_pi:
                        st.markdown(f"<div style='text-align:center; padding-top:8px; font-size:14px; color:#475569;'>Halaman: <b>{st.session_state.ps_current_page + 1} / {ps_total_pages}</b></div>", unsafe_allow_html=True)
                    with col_pn:
                        # MENGGUNAKAN CALLBACK (ON_CLICK) AGAR TIDAK PERLU ST.RERUN()
                        st.button("Next ➡️", key="ps_next_btn_2", use_container_width=True, disabled=(st.session_state.ps_current_page >= ps_total_pages - 1), on_click=next_page_2)

            # --- TAB 3: SUMMARIZATION ---
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
                            # Popover titik tiga diletakkan di sini agar melayang di pojok kanan atas kartu
                            with st.popover(""):
                                st.markdown("**Opsi Export**")
                
                                # Download TXT
                                st.download_button(
                                    "📄 Download TXT", 
                                    data=teks_hasil.encode('utf-8'), 
                                    file_name=f"sum_{target_sum_file}.txt", 
                                    use_container_width=True
                                )
                                # Download DOCX
                                try:
                                    from docx import Document
                                    doc_ex = Document()
                                    doc_ex.add_heading(f"Ringkasan: {target_sum_file}", 0)
                                    
                                    # Memecah teks berdasarkan dua baris baru untuk membuat paragraf asli
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
                                

                        # Garis pemisah halus di dalam kartu
                        st.markdown("<hr style='margin: 10px 0; opacity: 0.2;'>", unsafe_allow_html=True)

                        # Konten Teks Ringkasan (Gaya bersih, bukan blok kode abu-abu)
                        st.markdown(f"""
                        <div style='color: #334155; font-size: 16px; line-height: 1.8; text-align: justify; padding: 10px 5px;'>
                            {teks_hasil.replace('\n\n', '<br><br>')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        
                    # Aksi lanjutan (di luar kartu agar tidak menumpuk)
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
        fitur_nlp_dashboard(selected_files)
else:
    st.info("👋 Silakan upload file terlebih dahulu untuk mulai menggunakan dashboard.")
