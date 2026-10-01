"""Quick test of the symptom interpreter."""
import re, difflib, joblib

artifacts = joblib.load("disease_model.joblib")
symptom_columns = artifacts["symptom_columns"]

def format_name(s):
    return s.replace("_", " ").title()

symptom_display_map = {format_name(col): col for col in symptom_columns}
sorted_symptom_labels = sorted(symptom_display_map.keys())

_SYMPTOM_SYNONYMS = {
    "tired": "Fatigue", "stomach pain": "Sharp Abdominal Pain",
    "trouble breathing": "Difficulty Breathing", "headache": "Headache",
    "chest pain": "Sharp Chest Pain", "dizzy": "Dizziness",
    "itchy skin": "Itching Of Skin", "nauseous": "Nausea",
    "body ache": "Ache All Over", "coughing": "Cough",
    "sore throat": "Sore Throat", "phlegm": "Coughing Up Sputum",
    "chills": "Chills", "fever": "Fever",
}

def interpret_free_text_symptoms(text):
    if not text or not text.strip():
        return []
    text_clean = text.lower().strip()
    text_clean = re.sub(r'[,;\n]+', ' | ', text_clean)
    text_clean = re.sub(r'\band\b', ' | ', text_clean)
    phrases = [p.strip() for p in text_clean.split('|') if p.strip()]
    all_phrases = phrases + [text_clean.replace('|', ' ')]
    matched = {}
    symptom_labels_lower = {s.lower(): s for s in sorted_symptom_labels}
    synonyms_lower = {k.lower(): v for k, v in _SYMPTOM_SYNONYMS.items()}
    for phrase in all_phrases:
        phrase = phrase.strip()
        if len(phrase) < 2:
            continue
        if phrase in synonyms_lower:
            display = synonyms_lower[phrase]
            if display in symptom_display_map and display not in matched:
                matched[display] = {"source": phrase, "confidence": "high"}
                continue
        if phrase in symptom_labels_lower:
            display = symptom_labels_lower[phrase]
            if display not in matched:
                matched[display] = {"source": phrase, "confidence": "high"}
                continue
        for label_lower, label in symptom_labels_lower.items():
            if label in matched:
                continue
            if label_lower in phrase and len(label_lower) > 4:
                matched[label] = {"source": phrase, "confidence": "high"}
            elif phrase in label_lower and len(phrase) > 4:
                matched[label] = {"source": phrase, "confidence": "medium"}
        for syn_key, syn_display in synonyms_lower.items():
            if syn_display in matched:
                continue
            if syn_key in phrase and len(syn_key) > 3:
                if syn_display in symptom_display_map:
                    matched[syn_display] = {"source": phrase, "confidence": "medium"}
        close_matches = difflib.get_close_matches(phrase, list(symptom_labels_lower.keys()), n=2, cutoff=0.65)
        for cm in close_matches:
            display = symptom_labels_lower[cm]
            if display not in matched:
                matched[display] = {"source": phrase, "confidence": "low"}
    results = [{"matched": name, "source": info["source"], "confidence": info["confidence"]} for name, info in matched.items()]
    return results

tests = [
    "I have a bad headache, trouble breathing, stomach pain, feeling nauseous and very tired",
    "fever with chills, coughing, sore throat, body ache, dizzy, itchy skin",
    "my knee hurts and I have back pain with nausea",
]

for t in tests:
    results = interpret_free_text_symptoms(t)
    print(f"Input: {t}")
    print(f"Matched: {len(results)} symptoms")
    for r in results:
        print(f'  [{r["confidence"]:6s}] {r["matched"]} <- "{r["source"]}"')
    print()
