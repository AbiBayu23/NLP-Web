import stanza
import spacy.cli
import nltk

print("==================================================")
print("🚀 MEMULAI DOWNLOAD MODEL NLP...")
print("==================================================")

# 1. Download Model Stanza (Bahasa Indonesia)
print("\n[1/3] Mendownload Model Stanza (Bahasa Indonesia)...")
# Parameter 'verbose=True' agar kita bisa melihat progres bar di terminal
stanza.download('id', processors='tokenize,pos,lemma,depparse', verbose=True)

# 2. Download Model SpaCy (Bahasa Inggris & Lainnya)
# Kamu bisa menambahkan bahasa lain seperti 'fr_core_news_lg' di sini jika perlu
print("\n[2/3] Mendownload Model SpaCy (Bahasa Inggris dll)...")
spacy.cli.download("en_core_web_lg")
spacy.cli.download("es_core_news_lg") 
spacy.cli.download("fr_core_news_lg")
spacy.cli.download("de_core_news_lg")

# 3. Download Data Kamus NLTK
print("\n[3/3] Mendownload Dependensi NLTK (WordNet, VADER, dll)...")
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')

print("\n==================================================")
print("✅ SEMUA MODEL BERHASIL DIDOWNLOAD!")
print("Sekarang kamu bisa menjalankan aplikasi Streamlit.")
print("==================================================")
