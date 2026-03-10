import pandas as pd
from bs4 import BeautifulSoup
import re

# Fungsi natural_sort_key dan clean_duplicated_start dibiarkan SAMA PERSIS
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def clean_duplicated_start(source, target):
    s = source.strip()
    t = target.strip()
    if s.lower() == t.lower(): return t   
    words = s.split()
    if not words: return t
    pattern_str = r'\s+'.join(re.escape(w) for w in words)
    pattern = re.compile(r'^\s*' + pattern_str + r'\s*', re.IGNORECASE)
    if pattern.search(t):
        cleaned = pattern.sub('', t, count=1).strip()
        cleaned = re.sub(r'^[:,\-]\s*', '', cleaned)
        if cleaned: return cleaned   
    return t

# Fungsi yang disesuaikan untuk Streamlit
def process_eaf_untuk_streamlit(uploaded_file):
    # MEMBACA FILE DARI MEMORI STREAMLIT
    content = uploaded_file.getvalue().decode("utf-8")
    soup = BeautifulSoup(content, 'xml')

    parent_dict = {}
    for align in soup.find_all('ALIGNABLE_ANNOTATION'):
        p_id = align.get('ANNOTATION_ID')
        val = align.find('ANNOTATION_VALUE')
        if p_id and val:
            parent_dict[p_id] = val.text.strip()

    tier_data = {}
    for tier in soup.find_all('TIER'):
        t_id = tier.get('TIER_ID', '')
        tier_data[t_id] = {}
        for ref in tier.find_all('REF_ANNOTATION'):
            ref_id = ref.get('ANNOTATION_REF')
            val = ref.find('ANNOTATION_VALUE')
            if ref_id and val:
                word = val.text.strip()
                if ref_id in tier_data[t_id]:
                    tier_data[t_id][ref_id] = (tier_data[t_id][ref_id] + " " + word).replace("  ", " ")
                else:
                    tier_data[t_id][ref_id] = word

    rows = []
    for p_id, full_source in parent_dict.items():
        raw_targets = []
        catatan_texts = []
        for t_id, ref_dict in tier_data.items():
            if p_id not in ref_dict: continue
            text_val = ref_dict[p_id].strip()
            if "-note" in t_id.lower():
                catatan_texts.append(text_val)
            else:
                raw_targets.append(text_val)
        norm_source = re.sub(r'\s+', ' ', full_source.strip().lower())
        has_real_translation = any(re.sub(r'\s+', ' ', t.strip().lower()) != norm_source for t in raw_targets)

        filtered_targets = []
        for t in raw_targets:
            t_norm = re.sub(r'\s+', ' ', t.strip().lower())
            if t_norm == norm_source and has_real_translation: continue
            cleaned_t = clean_duplicated_start(full_source, t)
            if cleaned_t and cleaned_t not in filtered_targets:
                filtered_targets.append(cleaned_t)

        target_text = " ".join(filtered_targets).strip()
        target_text = clean_duplicated_start(full_source, target_text)
        catatan_text = " | ".join(catatan_texts).strip()

        if target_text or catatan_text:
            rows.append({
                "ID_Unit": p_id,
                "Source_Sentence": full_source,
                "Target_Sentence": target_text,
                "Catatan": catatan_text
            })

    rows.sort(key=lambda x: natural_sort_key(x['ID_Unit']))

    # MENGEMBALIKAN DATAFRAME, BUKAN SAVE KE EXCEL
    if rows:
        return pd.DataFrame(rows)
    else:
        return None