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

# ==========================================
# 1. KONFIGURASI HALAMAN WEB & TEMA
# ==========================================
st.set_page_config(
    page_title="AI NLP Dashboard", 
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
        
        /* Mempercantik fisik tombol selectbox */
        div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            cursor: pointer !important;
        }
        
        /* [PERBAIKAN] Trik Visual Aman: Menghilangkan kursor kedip tanpa merusak fungsi klik */
        div[data-baseweb="select"] input {
            caret-color: transparent !important; /* Menghilangkan kursor berkedip | / I-beam */
            cursor: pointer !important; /* Memaksa ikon mouse menjadi tangan (pointer) */
        }
        
        button[data-baseweb="tab"] {
            background-color: transparent !important;
        }
    </style>
    <div style='background-color:#E0F2FE; padding:20px; border-radius:12px; border-left: 8px solid #0EA5E9; margin-bottom:25px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
        <h2 style='color:#0369A1; margin:0; font-weight: 800;'>☁️ Explorer NLP Web App (Sky Edition)</h2>
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
        'pdf', 'fig', 'figure', 'table', 'vol', 'pp', 'ieee', 'al',
        'the', 'and', 'for', 'that'
    }
    words = [w for w in raw_words if len(w) > 2 and w not in stop_words_spacy and w not in daftar_hitam_kustom]
    word_counts = Counter(words).most_common(15)
    df_words = pd.DataFrame(word_counts, columns=['Kata', 'Frekuensi']) if word_counts else pd.DataFrame()
    
    return df_pos, df_words

DAFTAR_BAHASA = {
    'Indonesian': 'id', 'English': 'en', 'Spanish': 'es', 
    'French': 'fr', 'German': 'de', 'Japanese': 'ja', 'Korean': 'ko'
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

if 'local_files' not in st.session_state: st.session_state.local_files = {}
if 'summary_results' not in st.session_state: st.session_state.summary_results = {}

# ==========================================
# 3. UI UPLOAD & MANAJEMEN FILE
# ==========================================
st.markdown("<h3 style='color:#0F172A;'>📁 Analisis Dokumen Eksternal</h3>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("Upload File (PDF, DOCX, TXT)", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])

if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.local_files:
            with st.spinner(f"Menganalisis {file.name}..."):
                raw_text = extract_text(file)
                pola_kata = r'\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b'
                semua_kata = re.findall(pola_kata, raw_text.lower())
                
                kata_count = len(semua_kata)
                vocab_dokumen = set(semua_kata)
                
                doc_stats = nlp(raw_text[:80000]) 
                kalimat_count = len(list(doc_stats.sents))
                
                st.session_state.local_files[file.name] = {
                    'text': raw_text, 
                    'vocab': vocab_dokumen, 
                    'stats': {'k': kalimat_count, 'w': kata_count}
                }

if st.session_state.local_files:
    file_names = list(st.session_state.local_files.keys())
    col1, col2 = st.columns([8.5, 1.7])
    with col1: active_file = st.selectbox("Pilih Dokumen Aktif:", file_names)
    with col2:
        st.write("")
        st.write("")
        if st.button("🗑️ Hapus File", use_container_width=True):
            del st.session_state.local_files[active_file]
            st.rerun()

    if active_file in st.session_state.local_files:
        stats = st.session_state.local_files[active_file]['stats']
        st.markdown(f"""
        <div style='background-color:#F0F9FF; border: 1px solid #BAE6FD; padding:12px 20px; border-radius:8px; color:#0284C7; font-weight:600; margin-bottom: 20px;'>
            ✅ <b>Aktif:</b> {active_file} &nbsp;|&nbsp; 📜 {stats['k']} Kalimat &nbsp;|&nbsp; 🔤 {stats['w']} Kata
        </div>
        """, unsafe_allow_html=True)
        
        teks_dokumen = st.session_state.local_files[active_file]['text']
        
        # ==========================================
        # 4. VISUALISASI UTAMA
        # ==========================================
        st.markdown(f"<h3 style='color:#0F172A;'>📊 Gambaran Visual Dokumen</h3>", unsafe_allow_html=True)
        
        with st.expander("Klik di sini untuk Membuka / Menutup Visualisasi Data", expanded=False):
            with st.spinner("Menyiapkan grafik..."):
                df_pos, df_words = dapatkan_data_visual(teks_dokumen[:80000])
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.markdown("#### 📌 Statistik Tata Bahasa")
                    st.caption(f"Jumlah Kelas Kata dominan pada dokumen **{active_file}**")
                    if not df_pos.empty:
                        chart_pos = alt.Chart(df_pos).mark_bar(color="#0284C7", cornerRadiusEnd=4).encode(
                            y=alt.Y('POS Tag:N', sort='-x', title='Kelas Kata', 
                                    axis=alt.Axis(labelAngle=0, labelColor='#1E293B', labelFontWeight='normal', titleColor='#0F172A', grid=True, gridColor='#E2E8F0')), 
                            x=alt.X('Jumlah Kata:Q', title='Total Jumlah', 
                                    axis=alt.Axis(labelColor='#475569', titleColor='#0F172A', grid=True, gridColor='#CBD5E1', gridDash=[4,4])),
                            tooltip=['POS Tag', 'Jumlah Kata']
                        ).properties(height=380).configure_axis(
                            labelFontSize=13, titleFontSize=14, domainColor='#94A3B8', tickColor='#94A3B8'
                        ).configure_view(stroke='#94A3B8', strokeWidth=1)
                        st.altair_chart(chart_pos, use_container_width=True)
                
                with col_chart2:
                    st.markdown("#### 🔤 15 Kata Paling Sering Muncul")
                    st.caption(f"Kata yang sering mucul dalam dokumen **{active_file}**.")
                    if not df_words.empty:
                        chart_words = alt.Chart(df_words).mark_bar(color="#059669", cornerRadiusEnd=4).encode(
                            y=alt.Y('Kata', sort='-x', title='Kata Kunci', 
                                    axis=alt.Axis(labelAngle=0, labelColor='#1E293B', labelFontWeight='normal', titleColor='#0F172A', grid=True, gridColor='#E2E8F0')),
                            x=alt.X('Frekuensi', title='Jumlah Muncul', 
                                    axis=alt.Axis(labelColor='#475569', titleColor='#0F172A', grid=True, gridColor='#CBD5E1', gridDash=[4,4])),
                            tooltip=['Kata', 'Frekuensi']
                        ).properties(height=380).configure_axis(
                            labelFontSize=13, titleFontSize=14, domainColor='#94A3B8', tickColor='#94A3B8'
                        ).configure_view(stroke='#94A3B8', strokeWidth=1)
                        st.altair_chart(chart_words, use_container_width=True)
        
        st.write("---") 
        
        # ==========================================
        # 5. TAB FITUR NLP 
        # ==========================================
        tab_search, tab_pos_search, tab_summary = st.tabs([
            "🔍 Keyword Search", "🕵️‍♂️ POS Search", "📝 Summarization"
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
                    # [PERBAIKAN DINAMIS] Menghitung SEMUA jenis tata bahasa yang ada di kalimat
                    counts = Counter([t.pos_ for t in doc_m if t.pos_ not in ['SPACE', 'PUNCT', 'SYM', 'X']])
                    
                    # Highlight Logic
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

                    # Build dynamic pills
                    pills_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>"
                    for pos in sorted(counts.keys()):
                        bg = Warna_POS_Utama.get(pos, '#94A3B8')
                        pills_html += f"<span style='background-color: {bg}; color:white; border-radius:4px; padding:2px 8px; font-size:11px; font-weight:600;'>{pos}: {counts[pos]}</span>"
                    pills_html += "</div>"

                    st.markdown(f"""<div style='background:#FFFFFF; padding:20px; border-radius:10px; border:1px solid #E2E8F0; margin-bottom:10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                        <div style='color:#334155; font-size:15px; margin-bottom:20px; line-height:1.7;'>{highlighted}</div>
                        <div style='border-top:1px solid #F1F5F9; padding-top:12px; display:flex; justify-content:space-between; align-items:center;'>
                            <div style='font-size:11px; color:#0284C7; font-weight:bold; background:#F0F9FF; padding:3px 8px; border-radius:4px;'>📄 DOKUMEN: {active_file.upper()}</div>
                            <div>{pills_html}</div>
                        </div></div>""", unsafe_allow_html=True)
                    
                    if aksi == "🏷️ POS Tag":
                        st.markdown(get_colored_pos_text(match), unsafe_allow_html=True); st.write("")
                    elif aksi == "🌐 Trans":
                        l1, l2, _ = st.columns([3, 1, 6])
                        with l1: target_lang = st.selectbox("Ke:", list(DAFTAR_BAHASA.keys()), key=f"tr_{i}_{start}", label_visibility="collapsed")
                        with l2: go = st.button("Ok", key=f"go_{i}_{start}", use_container_width=True)
                        if go:
                            res = GoogleTranslator(source='auto', target=DAFTAR_BAHASA[target_lang]).translate(match)
                            st.markdown(f"<div style='background:#E0F2FE; border: 1px solid #7DD3FC; padding:15px; color:#0369A1; border-radius:8px; font-size:15px; margin-bottom:15px;'>{res}</div>", unsafe_allow_html=True)

        # --- TAB 1: SEARCHING ---
        with tab_search:
            st.markdown("<h3 style='color:#0F172A;'>🔍 Pencarian Pintar (Lemmatization)</h3>", unsafe_allow_html=True)
            
            if 'teks_pencarian' not in st.session_state: st.session_state.teks_pencarian = ""
            if 'query_lemma' not in st.session_state: st.session_state.query_lemma = "" 
            if 'trigger_search' not in st.session_state: st.session_state.trigger_search = False
            if 'search_results' not in st.session_state: st.session_state.search_results = []
            if 'current_page' not in st.session_state: st.session_state.current_page = 0
            
            def aksi_klik_saran():
                pilihan = st.session_state.get('saran_pills_widget')
                if pilihan:
                    st.session_state.teks_pencarian = pilihan
                    st.session_state.trigger_search = True
                    st.session_state.current_page = 0

            col_input, col_btn = st.columns([7, 1], gap="small")
            with col_input:
                query_aktif = st.text_input("Pencarian", key="teks_pencarian", placeholder="Ketik kata kunci (misal: analysis)...", label_visibility="collapsed")
            with col_btn:
                btn_cari = st.button("Cari", use_container_width=True, type="primary")

            if query_aktif:
                sinonim_set = set()
                try:
                    for syn in wordnet.synsets(query_aktif):
                        for l in syn.lemmas():
                            kata_terkait = l.name().lower()
                            if kata_terkait != query_aktif.lower() and '_' not in kata_terkait:
                                sinonim_set.add(kata_terkait)
                except: pass
                
                vocab_aktif = st.session_state.local_files[active_file]['vocab']
                saran_kata = [kata for kata in sinonim_set if kata in vocab_aktif][:8] 
                
                if saran_kata:
                    st.markdown("<div style='font-size:14px; color:#64748B; margin-bottom:8px;'>💡 Saran kata terkait yang <b>ada di dokumen ini</b>:</div>", unsafe_allow_html=True)
                    try:
                        st.pills("Saran", saran_kata, key="saran_pills_widget", on_change=aksi_klik_saran, label_visibility="collapsed")
                    except AttributeError:
                        pass

            if btn_cari or st.session_state.trigger_search:
                st.session_state.trigger_search = False 
                if st.session_state.teks_pencarian:
                    query_doc = nlp(st.session_state.teks_pencarian.strip())
                    query_lemma = query_doc[0].lemma_.lower() if len(query_doc) > 0 else st.session_state.teks_pencarian.lower()
                    st.session_state.query_lemma = query_lemma
                    
                    doc = nlp(teks_dokumen[:100000])
                    matches = []
                    for s in doc.sents:
                        if any(token.lemma_.lower() == query_lemma for token in s):
                            matches.append(s.text.strip())
                            
                    st.session_state.search_results = matches
                    st.session_state.current_page = 0

            if st.session_state.search_results:
                total_results = len(st.session_state.search_results)
                
                col_info, col_blank, col_filter_text, col_filter_drop = st.columns([5, 1, 3, 1.6], gap="small")
                with col_info:
                    st.markdown(f"<div style='color:#059669; font-weight:bold; font-size:14.5px; padding-top:8px;'>✅ Ditemukan {total_results} baris kalimat.</div>", unsafe_allow_html=True)
                with col_filter_text:
                    st.markdown("<div style='text-align:right; padding-top:8px; font-size:14px; color:#475569; font-weight:00;'>Tampilkan per halaman:</div>", unsafe_allow_html=True)
                with col_filter_drop:
                    opsi_limit = ["5", "10", "25", "50", "100", "All"]
                    pilihan_limit = st.selectbox("Limit", opsi_limit, key="limit_search", label_visibility="collapsed")
                
                if pilihan_limit == "All":
                    ITEMS_PER_PAGE = total_results if total_results > 0 else 1
                else:
                    ITEMS_PER_PAGE = int(pilihan_limit)
                    
                total_pages = max(1, math.ceil(total_results / ITEMS_PER_PAGE))
                
                if st.session_state.current_page >= total_pages:
                    st.session_state.current_page = max(0, total_pages - 1)
                    
                start_idx = st.session_state.current_page * ITEMS_PER_PAGE
                end_idx = start_idx + ITEMS_PER_PAGE
                subset_results = st.session_state.search_results[start_idx:end_idx]

                st.write("") 

                for i, match in enumerate(subset_results):
                    col_card, col_action = st.columns([8.5, 1.5], gap="small")
                    
                    with col_action:
                        aksi = st.selectbox("Aksi", ["Aksi", "🏷️ POS Tag", "🌐 Trans"], key=f"aksi_{start_idx + i}", label_visibility="collapsed")
                    
                    with col_card:
                        doc_match = nlp(match)
                        kata_cocok = set([token.text for token in doc_match if token.lemma_.lower() == st.session_state.query_lemma])
                        if kata_cocok:
                            pola_regex = r"\b(" + "|".join(map(re.escape, kata_cocok)) + r")\b"
                            highlighted = re.sub(pola_regex, r"<mark style='background:#FDE047; color:#0F172A; font-weight:bold; padding:0 4px; border-radius:3px;'>\1</mark>", match, flags=re.I)
                        else:
                            highlighted = match
                            
                        pos_counts = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'PRON': 0, 'ADP': 0, 'PROPN': 0}
                        for token in doc_match:
                            if token.pos_ in pos_counts: pos_counts[token.pos_] += 1
                                
                        pills_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>"
                        for pos, count in pos_counts.items():
                            if count > 0:
                                bg_color = Warna_POS_Utama.get(pos, '#94A3B8')
                                pills_html += f"<span style='background-color: {bg_color}; color:white; border-radius:4px; padding:2px 8px; font-size:11px; font-weight:600;'>{pos.capitalize()}: {count}</span>"
                        pills_html += "</div>"
                        
                        html_card = f"""
                        <div style='background:#FFFFFF; padding:20px; border-radius:10px; border:1px solid #E2E8F0; margin-bottom:10px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); min-height:90px;'>
                            <div style='color:#334155; font-size:15px; margin-bottom:20px; line-height:1.7;'>{highlighted}</div>
                            <div style='border-top:1px solid #F1F5F9; padding-top:12px; display:flex; justify-content:space-between; align-items:center;'>
                                <div style='font-size:11px; color:#0284C7; font-weight:bold; letter-spacing: 0.5px; background:#F0F9FF; padding:3px 8px; border-radius:4px;'>📄 DOKUMEN: {active_file.upper()}</div>
                                <div>{pills_html}</div>
                            </div>
                        </div>
                        """
                        st.markdown(html_card, unsafe_allow_html=True)
                        
                        if aksi == "🏷️ POS Tag":
                            st.markdown(get_colored_pos_text(match), unsafe_allow_html=True)
                            st.write("")
                        elif aksi == "🌐 Trans":
                            col_lang, col_go, _ = st.columns([3, 1, 6])
                            with col_lang:
                                target_lang_name = st.selectbox("Ke:", list(DAFTAR_BAHASA.keys()), key=f"lang_{start_idx + i}", label_visibility="collapsed")
                            with col_go:
                                go_trans = st.button("Ok", key=f"go_{start_idx + i}", use_container_width=True)
                            
                            if go_trans:
                                target_lang_code = DAFTAR_BAHASA[target_lang_name]
                                with st.spinner("Menerjemahkan..."):
                                    try:
                                        hasil_terjemahan = GoogleTranslator(source='auto', target=target_lang_code).translate(match)
                                        st.markdown(f"<div style='background:#E0F2FE; border: 1px solid #7DD3FC; padding:15px; color:#0369A1; border-radius:8px; font-size:15px; margin-bottom:15px; font-weight:500;'>{hasil_terjemahan}</div>", unsafe_allow_html=True)
                                    except Exception as e: st.error(f"Error: {e}")

                st.write("") 
                col_blank1, col_p, col_page_info, col_n, col_blank2 = st.columns([2, 1, 2, 1, 2])
                with col_p:
                    if st.button("⬅️ Prev", use_container_width=True, disabled=(st.session_state.current_page == 0)):
                        st.session_state.current_page -= 1; st.rerun()
                with col_page_info:
                    st.markdown(f"<div style='text-align:center; padding-top:8px; font-size:14px; color:#475569;'>Halaman: <b>{st.session_state.current_page + 1} / {total_pages}</b></div>", unsafe_allow_html=True)
                with col_n:
                    if st.button("Next ➡️", use_container_width=True, disabled=(st.session_state.current_page >= total_pages - 1)):
                        st.session_state.current_page += 1; st.rerun()

        # --- TAB 2: POS SEARCH ---
        with tab_pos_search:
            st.markdown("<h3 style='color:#0F172A;'>🕵️‍♂️ Pencarian Spesifik (POS Search)</h3>", unsafe_allow_html=True)
            st.caption("Bisa diisi salah satu (hanya kelas kata, atau hanya kata kunci), atau isi dua-duanya untuk pencarian presisi maksimal.")
            
            if 'ps_query' not in st.session_state: st.session_state.ps_query = ""
            if 'ps_tag' not in st.session_state: st.session_state.ps_tag = "ALL"
            if 'ps_results' not in st.session_state: st.session_state.ps_results = []
            if 'ps_current_page' not in st.session_state: st.session_state.ps_current_page = 0
            
            col_ps1, col_ps2, col_ps3 = st.columns([4, 4, 1.4], gap="small")
            with col_ps1:
                ps_keyword = st.text_input("Kata Kunci (Opsional):", placeholder="Ketik kata (misal: BI-LSTM)...", key="input_ps_keyword")
            with col_ps2:
                ps_target_tag = st.selectbox("Sebagai Kelas Kata (Opsional):", 
                    ["SEMUA KELAS KATA", "VERB (Kata Kerja)", "NOUN (Kata Benda)", "ADJ (Kata Sifat)", "ADV (Kata Keterangan)", "PROPN (Nama/Entitas)"], 
                    key="input_ps_tag"
                )
            with col_ps3:
                st.write("") 
                st.write("")
                btn_ps_cari = st.button("Cari Presisi", type="primary", use_container_width=True)
                
            if btn_ps_cari:
                if not ps_keyword.strip() and ps_target_tag == "SEMUA KELAS KATA":
                    st.warning("⚠️ Masukkan kata kunci ATAU pilih kelas kata terlebih dahulu.")
                else:
                    map_tags = {"SEMUA KELAS KATA": "ALL", "VERB (Kata Kerja)": "VERB", "NOUN (Kata Benda)": "NOUN", "ADJ (Kata Sifat)": "ADJ", "ADV (Kata Keterangan)": "ADV", "PROPN (Nama/Entitas)": "PROPN"}
                    target_tag = map_tags[ps_target_tag]
                    
                    st.session_state.ps_query = ps_keyword.strip()
                    st.session_state.ps_tag = target_tag
                
                    q_lemma = ""
                    if st.session_state.ps_query:
                        q_doc = nlp(st.session_state.ps_query)
                        q_lemma = q_doc[0].lemma_.lower() if len(q_doc) > 0 else st.session_state.ps_query.lower()
                    
                    doc = nlp(teks_dokumen[:100000])
                    ps_matches = []
                    
                    for s in doc.sents:
                        match_found = False
                        for token in s:
                            match_kw = (token.lemma_.lower() == q_lemma) if q_lemma else True
                            match_pos = (token.pos_ == target_tag) if target_tag != "ALL" else True
                            
                            if match_kw and match_pos:
                                match_found = True
                                break
                        
                        if match_found:
                            ps_matches.append(s.text.strip())
                            
                    st.session_state.ps_results = ps_matches
                    st.session_state.ps_current_page = 0
            
            if st.session_state.ps_results:
                ps_total = len(st.session_state.ps_results)
                
                if st.session_state.ps_query and st.session_state.ps_tag != "ALL":
                    info_msg = f"✅ Ditemukan {ps_total} kalimat di mana '{st.session_state.ps_query}' berperan sebagai **{st.session_state.ps_tag}**."
                elif st.session_state.ps_query:
                    info_msg = f"✅ Ditemukan {ps_total} kalimat yang mengandung kata '{st.session_state.ps_query}'."
                else:
                    info_msg = f"✅ Ditemukan {ps_total} kalimat yang memiliki struktur **{st.session_state.ps_tag}**."
                    
                col_info_ps, col_blank_ps, col_filter_text_ps, col_filter_drop_ps = st.columns([6, 0.5, 3, 2], gap="small")
                with col_info_ps:
                    st.markdown(f"<div style='color:#059669; font-weight:bold; font-size:14.5px; padding-top:8px;'>{info_msg}</div>", unsafe_allow_html=True)
                with col_filter_text_ps:
                    st.markdown("<div style='text-align:right; padding-top:8px; font-size:14px; color:#475569; font-weight:600;'>Tampilkan per halaman:</div>", unsafe_allow_html=True)
                with col_filter_drop_ps:
                    ps_opsi_limit = ["5", "10", "25", "50", "100", "All"]
                    ps_pilihan_limit = st.selectbox("Limit PS", ps_opsi_limit, key="limit_ps_search", label_visibility="collapsed")
                
                if ps_pilihan_limit == "All":
                    PS_ITEMS_PER_PAGE = ps_total if ps_total > 0 else 1
                else:
                    PS_ITEMS_PER_PAGE = int(ps_pilihan_limit)
                    
                ps_total_pages = max(1, math.ceil(ps_total / PS_ITEMS_PER_PAGE))
                
                if st.session_state.ps_current_page >= ps_total_pages:
                    st.session_state.ps_current_page = max(0, ps_total_pages - 1)
                    
                ps_start = st.session_state.ps_current_page * PS_ITEMS_PER_PAGE
                ps_end = ps_start + PS_ITEMS_PER_PAGE
                ps_subset = st.session_state.ps_results[ps_start:ps_end]
                
                st.write("") 

                for j, match_text in enumerate(ps_subset):
                    col_ps_card, col_ps_act = st.columns([8.5, 1.5], gap="small")
                    
                    with col_ps_act:
                        ps_aksi = st.selectbox("Aksi", ["Aksi", "🏷️ POS Tag", "🌐 Trans"], key=f"ps_aksi_{ps_start + j}", label_visibility="collapsed")
                    
                    with col_ps_card:
                        doc_match = nlp(match_text)
                        
                        q_lemma2 = ""
                        if st.session_state.ps_query:
                            q_doc2 = nlp(st.session_state.ps_query)
                            q_lemma2 = q_doc2[0].lemma_.lower() if len(q_doc2) > 0 else st.session_state.ps_query.lower()
                        
                        highlighted_html = ""
                        pos_counts = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'PRON': 0, 'ADP': 0, 'PROPN': 0}
                        
                        for token in doc_match:
                            if token.pos_ in pos_counts: pos_counts[token.pos_] += 1
                            
                            match_kw = (token.lemma_.lower() == q_lemma2) if q_lemma2 else True
                            match_pos = (token.pos_ == st.session_state.ps_tag) if st.session_state.ps_tag != "ALL" else True
                            
                            if match_kw and match_pos:
                                highlighted_html += f"<mark style='background:#EF4444; color:white; font-weight:bold; padding:0 4px; border-radius:4px; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>{token.text}</mark>{token.whitespace_}"
                            else:
                                highlighted_html += f"{token.text}{token.whitespace_}"
                                
                        pills_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>"
                        for pos, count in pos_counts.items():
                            if count > 0:
                                bg_color = Warna_POS_Utama.get(pos, '#94A3B8')
                                pills_html += f"<span style='background-color: {bg_color}; color:white; border-radius:4px; padding:2px 8px; font-size:11px; font-weight:600;'>{pos.capitalize()}: {count}</span>"
                        pills_html += "</div>"
                        
                        tanda_pos = st.session_state.ps_tag if st.session_state.ps_tag != "ALL" else "BEBAS"
                        
                        html_card = f"""
                        <div style='background:#FFFFFF; padding:20px; border-radius:10px; border:1px solid #E2E8F0; margin-bottom:10px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); min-height:90px;'>
                            <div style='color:#334155; font-size:14.5px; margin-bottom:20px; line-height:1.7;'>{highlighted_html}</div>
                            <div style='border-top:1px solid #F1F5F9; padding-top:12px; display:flex; justify-content:space-between; align-items:center;'>
                                <div style='font-size:11px; color:#EF4444; font-weight:bold; letter-spacing: 0.5px; background:#FEF2F2; padding:3px 8px; border-radius:4px;'>🎯 POS TARGET: {tanda_pos}</div>
                                <div>{pills_html}</div>
                            </div>
                        </div>
                        """
                        st.markdown(html_card, unsafe_allow_html=True)
                        
                        if ps_aksi == "🏷️ POS Tag":
                            st.markdown(get_colored_pos_text(match_text), unsafe_allow_html=True)
                            st.write("")
                        elif ps_aksi == "🌐 Trans":
                            col_lang, col_go, _ = st.columns([3, 1, 6])
                            with col_lang:
                                t_lang_name = st.selectbox("Ke:", list(DAFTAR_BAHASA.keys()), key=f"ps_lang_{ps_start + j}", label_visibility="collapsed")
                            with col_go:
                                ps_go_trans = st.button("Ok", key=f"ps_go_{ps_start + j}", use_container_width=True)
                            
                            if ps_go_trans:
                                t_lang_code = DAFTAR_BAHASA[t_lang_name]
                                with st.spinner("Menerjemahkan..."):
                                    try:
                                        h_trans = GoogleTranslator(source='auto', target=t_lang_code).translate(match_text)
                                        st.markdown(f"<div style='background:#E0F2FE; border: 1px solid #7DD3FC; padding:15px; color:#0369A1; border-radius:8px; font-size:15px; margin-bottom:15px; font-weight:500;'>{h_trans}</div>", unsafe_allow_html=True)
                                    except Exception as e: st.error(f"Error: {e}")

                st.write("") 
                col_b1, col_pp, col_pi, col_pn, col_b2 = st.columns([2, 1, 2, 1, 2])
                with col_pp:
                    if st.button("⬅️ Prev", key="ps_prev", use_container_width=True, disabled=(st.session_state.ps_current_page == 0)):
                        st.session_state.ps_current_page -= 1; st.rerun()
                with col_pi:
                    st.markdown(f"<div style='text-align:center; padding-top:8px; font-size:14px; color:#475569;'>Halaman: <b>{st.session_state.ps_current_page + 1} / {ps_total_pages}</b></div>", unsafe_allow_html=True)
                with col_pn:
                    if st.button("Next ➡️", key="ps_next", use_container_width=True, disabled=(st.session_state.ps_current_page >= ps_total_pages - 1)):
                        st.session_state.ps_current_page += 1; st.rerun()

        # --- TAB 3: SUMMARIZATION ---
        with tab_summary:
            st.markdown("<h3 style='color:#0F172A;'>📝 Ekstraksi Dokumen Cepat (LexRank)</h3>", unsafe_allow_html=True)
            st.caption("Algoritma ini memindai secara cerdas dan menyusun poin-poin paling vital dari file Anda menjadi paragraf yang rapi dan padat.")
            
            if st.button("🚀 Mulai Ekstraksi Kilat", type="primary"):
                with st.spinner(f"Mengekstrak informasi penting dari dokumen '{active_file}'..."):
                    try:
                        total_kalimat_dokumen = stats['k']
                        target_jumlah_kalimat = max(5, min(int(total_kalimat_dokumen * 0.15), 30))
                        
                        parser = PlaintextParser.from_string(teks_dokumen, Tokenizer("english"))
                        summarizer_cepat = LexRankSummarizer()
                        hasil_ekstraksi = summarizer_cepat(parser.document, target_jumlah_kalimat)
                        
                        kalimat_terekstrak = [str(sentence) for sentence in hasil_ekstraksi]
                        
                        kalimat_per_paragraf = 4
                        list_paragraf = []
                        for i in range(0, len(kalimat_terekstrak), kalimat_per_paragraf):
                            paragraf = " ".join(kalimat_terekstrak[i:i+kalimat_per_paragraf])
                            list_paragraf.append(paragraf)
                            
                        html_ringkasan = "<br><br>".join(list_paragraf)
                        st.session_state.summary_results[active_file] = html_ringkasan
                        
                        st.success(f"✅ Ekstraksi selesai! Berhasil merangkum '{active_file}' ({total_kalimat_dokumen} kalimat) menjadi {len(kalimat_terekstrak)} kalimat inti.")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat mengekstrak: {e}")

            if getattr(st.session_state, 'summary_results', None) and active_file in st.session_state.summary_results:
                teks_html_ringkasan = st.session_state.summary_results[active_file]
                
                teks_bersih_untuk_hitung = teks_html_ringkasan.replace("<br><br>", " ")
                jumlah_kata_sum = len(re.findall(r'\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b', teks_bersih_untuk_hitung))
                
                doc_sum = nlp(teks_bersih_untuk_hitung)
                pos_counts_sum = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'PRON': 0, 'ADP': 0, 'PROPN': 0}
                for token in doc_sum:
                    if token.pos_ in pos_counts_sum:
                        pos_counts_sum[token.pos_] += 1
                
                st.markdown(f"<h3 style='color:#0F172A; margin-top:30px;'>📑 Hasil Ekstraksi: <b>{active_file}</b></h3>", unsafe_allow_html=True)
                
                pills_html_sum = "<div style='display:flex; gap:8px; flex-wrap:wrap; margin-bottom: 15px;'>"
                pills_html_sum += f"<span style='border:1px solid #0EA5E9; color:#0369A1; border-radius:5px; padding:6px 14px; font-size:13px; font-weight:bold; background-color: #E0F2FE; box-shadow: 0 1px 2px rgba(0,0,0,0.05);'>🔤 Total Kata: {jumlah_kata_sum}</span>"
                
                for pos, count in pos_counts_sum.items():
                    if count > 0:
                        bg_color = Warna_POS_Utama.get(pos, '#94A3B8')
                        pills_html_sum += f"<span style='background-color: {bg_color}; color:white; border-radius:5px; padding:5px 12px; font-size:12.5px; font-weight:600; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>{pos}: {count}</span>"
                pills_html_sum += "</div>"
                
                st.markdown(pills_html_sum, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style='background-color: #FFFFFF; color: #334155; padding: 30px; border-radius: 12px; font-size: 16px; line-height: 1.8; text-align: justify; border: 1px solid #BAE6FD; margin-bottom: 25px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);'>
                    {teks_html_ringkasan}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<h4 style='color:#0F172A;'>🛠️ Aksi pada Hasil Ringkasan</h4>", unsafe_allow_html=True)
                aksi_ringkasan = st.selectbox("Pilih Aksi:", ["Pilih Aksi...", "🏷️ POS Tagging", "🌐 Translate"], key="aksi_sum_utama")
                
                if aksi_ringkasan == "🏷️ POS Tagging":
                    with st.spinner("Membedah struktur kata ringkasan..."):
                        st.markdown(get_colored_pos_text(teks_bersih_untuk_hitung), unsafe_allow_html=True)
                        
                elif aksi_ringkasan == "🌐 Translate":
                    col_lang_sum, col_go_sum, _ = st.columns([2, 1, 5])
                    with col_lang_sum:
                        target_lang_sum_name = st.selectbox("Terjemahkan ke:", list(DAFTAR_BAHASA.keys()), key="lang_sum_utama", label_visibility="collapsed")
                    with col_go_sum:
                        go_trans_sum = st.button("Ok", key="go_sum_utama", use_container_width=True)
                        
                    if go_trans_sum:
                        target_lang_sum_code = DAFTAR_BAHASA[target_lang_sum_name]
                        with st.spinner(f"Menerjemahkan ringkasan {active_file}..."):
                            try:
                                teks_terjemah = teks_html_ringkasan.replace("<br><br>", " \n\n ")
                                hasil_terjemahan_sum = GoogleTranslator(source='auto', target=target_lang_sum_code).translate(teks_terjemah[:4000])
                                hasil_terjemahan_html = hasil_terjemahan_sum.replace(" \n\n ", "<br><br>")
                                
                                st.markdown(f"""
                                <div style='background-color: #F0FDF4; color: #065F46; padding: 25px; border-radius: 12px; font-size: 16px; line-height: 1.8; text-align: justify; border: 1px solid #A7F3D0; margin-top: 15px; box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.03);'>
                                    {hasil_terjemahan_html}
                                </div>
                                """, unsafe_allow_html=True)
                            except Exception as e: 
                                st.error(f"Error: {e}")
else:
    st.info("👋 Silakan upload file terlebih dahulu untuk mulai menggunakan dashboard.")

