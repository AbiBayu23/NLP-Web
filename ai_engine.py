

# ==========================================
# FILE: ai_engine.py
# (Mesin Pemroses AI dan NLP)
# ==========================================
import streamlit as st
import spacy
from spacy.language import Language
import re

# --- KAMUS POS AKADEMIS INDONESIA ---
KAMUS_ADP = {"di", "ke", "dari", "pada", "dalam", "dengan", "bagi", "untuk", "kepada", "daripada", "oleh", "tentang", "seperti", "beserta", "secara", "melalui", "menuju", "antara", "demi", "hingga", "sampai", "sebagai"}
KAMUS_CCONJ = {"dan", "atau", "tetapi", "melainkan", "sedangkan", "serta", "lalu", "kemudian"}
KAMUS_SCONJ = {"karena", "jika", "kalau", "bahwa", "ketika", "sejak", "agar", "supaya", "meskipun", "walaupun", "sehingga", "sebab", "sementara", "apabila", "biarpun"}
KAMUS_AUX = {"adalah", "ialah", "merupakan", "akan", "sedang", "telah", "sudah", "belum", "dapat", "bisa", "boleh", "harus", "wajib", "mungkin", "pasti", "pernah", "ingin", "mau"}
KAMUS_PRON = {"saya", "aku", "kita", "kami", "kamu", "engkau", "anda", "dia", "ia", "mereka", "beliau", "apa", "siapa", "mana", "seseorang", "sesuatu"}
KAMUS_DET = {"ini", "itu", "tersebut", "suatu", "sebuah", "sang", "para", "tiap", "setiap", "beberapa", "berbagai", "segala", "semua", "seluruh"}
KAMUS_ADV = {"sangat", "paling", "lebih", "hanya", "saja", "amat", "nian", "sekali", "tidak", "bukan", "jangan", "selalu", "sering", "kadang", "segera", "cukup", "kurang", "sekadar", "sekedar"}
KAMUS_NOUN_AKADEMIS = {"pendahuluan", "bab", "transformasi", "riset", "konsumen", "akurasi", "alasan", "model", "pemrosesan", "data", "bahasa", "perangkat", "periset", "paradigma", "analisis", "lanskap", "metode", "metodologi", "hasil", "kesimpulan", "tujuan", "latar", "belakang", "masalah", "pustaka", "tinjauan", "teori", "implementasi", "sistem", "aplikasi", "pengguna", "evaluasi", "pengujian", "kesalahan", "nilai", "tabel", "gambar", "grafik", "penelitian", "pengembangan", "studi", "pendekatan", "kinerja", "parameter", "arsitektur", "algoritma", "akurasi", "presisi", "informasi"}
KAMUS_VERB_AKADEMIS = {"menggunakan", "mendukung", "memastikan", "menyaksikan", "bersifat", "melakukan", "membuat", "merancang", "mengimplementasikan", "menganalisis", "membahas", "menyimpulkan", "menunjukkan", "berdasarkan", "bertujuan", "meningkatkan", "menurunkan", "mengevaluasi", "membandingkan", "menghasilkan", "ditemukan"}
KAMUS_ADJ_AKADEMIS = {"multilingual", "utama", "global", "presisi", "manual", "modern", "konfirmatori", "krusial", "penting", "baru", "lama", "baik", "buruk", "tinggi", "rendah", "besar", "kecil", "signifikan", "otomatis", "statis", "dinamis", "akurat", "efektif", "efisien", "relevan", "spesifik", "umum", "kompleks", "sederhana"}

SPACY_MODELS = {
    'id': 'stanza', 'en': 'en_core_web_lg', 'es': 'es_core_news_lg',
    'fr': 'fr_core_news_lg', 'de': 'de_core_news_lg'
}

# --- PIPELINE KUSTOM ---
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

@Language.component("koreksi_pos_id")
def koreksi_pos_id(doc):
    for token in doc:
        kata_lower = token.text.lower()
        if kata_lower in KAMUS_ADP: token.pos_ = "ADP"
        elif kata_lower in KAMUS_CCONJ: token.pos_ = "CCONJ"
        elif kata_lower in KAMUS_SCONJ: token.pos_ = "SCONJ"
        elif kata_lower in KAMUS_AUX: token.pos_ = "AUX"
        elif kata_lower in KAMUS_PRON: token.pos_ = "PRON"
        elif kata_lower in KAMUS_DET: token.pos_ = "DET"
        elif kata_lower in KAMUS_ADV: token.pos_ = "ADV"
        elif kata_lower in KAMUS_NOUN_AKADEMIS: token.pos_ = "NOUN"
        elif kata_lower in KAMUS_VERB_AKADEMIS: token.pos_ = "VERB"
        elif kata_lower in KAMUS_ADJ_AKADEMIS: token.pos_ = "ADJ"
    return doc

# --- FUNGSI LOAD NLP ---
@st.cache_resource
def load_ai_model(lang_code):
    model_name = SPACY_MODELS.get(lang_code, 'en_core_web_sm') 
    
    if model_name == 'stanza':
        import spacy_stanza
        # Pastikan tidak ada perintah download() di sini
        nlp_model = spacy_stanza.load_pipeline('id', processors='tokenize,pos,lemma,depparse')
        if "koreksi_pos_id" not in nlp_model.pipe_names:
            nlp_model.add_pipe("koreksi_pos_id", last=True)
        return nlp_model
            
    # Untuk SpaCy LG
    try:
        nlp_model = spacy.load(model_name)
    except Exception:
        # Fallback ke model dasar jika LG benar-benar tidak ada
        from spacy.lang.en import English
        nlp_model = English()
        nlp_model.add_pipe("sentencizer")
        
    if "merge_hyphens" not in nlp_model.pipe_names:
        # Tambahkan pipeline kustom merger
        nlp_model.add_pipe("merge_hyphens", first=True)
        
    return nlp_model

# --- FUNGSI LOAD AUDIO WHISPER ---
@st.cache_resource
def load_whisper_model():
    import whisper
    return whisper.load_model("base")

# --- FUNGSI LOAD SINONIM WORDNET ---
@st.cache_data(show_spinner=False)
def dapatkan_sinonim(query, lang_code):
    from nltk.corpus import wordnet
    try:
        wordnet.ensure_loaded()
    except Exception:
        pass
        
    wn_lang = 'ind' if lang_code == 'id' else ('fra' if lang_code == 'fr' else ('spa' if lang_code == 'es' else 'eng'))
    sinonim_set = set([query.lower()])
    try:
        for syn in wordnet.synsets(query, lang=wn_lang):
            for l in syn.lemmas(lang=wn_lang): 
                kata_terkait = l.name().lower().replace('_', ' ')
                sinonim_set.add(kata_terkait)
    except Exception:
        pass
    return sinonim_set

# --- FUNGSI LOAD GEMINI (UNTUK COMPARE INSIGHT) ---
