import pandas as pd
import spacy
import os
from lexical_diversity import lex_div as ld

nlp = spacy.load("es_core_news_lg")

def load_file_as_dataframe(str_file_directory, enc='utf8', dlm=','):
    print('Loading file')
    df = pd.read_csv(str_file_directory, encoding=enc, delimiter=dlm)
    print(f'File loaded: {df.shape[0]} rows, {df.shape[1]} columns')
    print('\nFirst two columns:\n')
    print(df.iloc[:, :2].head())
    print(df.shape)
    return df

def spacy_analysis_to_df(df, text_col):
    rows = []

    for idx, text in df[text_col].items():
        doc = nlp(text if isinstance(text, str) else "")
        for token in doc:
            rows.append({
                "row_index": idx,
                "original_text": text,
                "token": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "fine_pos": token.tag_,
                "morphology": str(token.morph),
                "dep_labels": token.dep_,
                "head": token.head.text,
                "has_vector": token.has_vector,
                "vector_norm": token.vector_norm,
                "sent_text": token.sent.text,
            })

    return pd.DataFrame(rows)

def basic_characteristics(task_id, prompt):
    doc = nlp(prompt)
    tokens = len([t for t in doc])
    words = len([t for t in doc if t.is_alpha])
    sentences = len(list(doc.sents))
    return {
        'task_ID': task_id,
        'tokens': tokens,
        'words': words,
        'sentences': sentences
    }

def mtld(tokens, ttr_threshold=0.72, min_tokens=3):
    """
    Calculate MTLD (Measure of Textual Lexical Diversity).
    tokens: list of strings (e.g. lemmas or words)
    ttr_threshold: the TTR value at which to start a new factor
    min_tokens: minimum number of tokens to compute MTLD
    Returns: float (MTLD score)
    """
    if len(tokens) < min_tokens:
        return float(len(tokens))  # fallback or warning: too short

    def count_factors(tokens):
        types = set()
        token_count = 0
        type_count = 0
        ttr = 1.0
        factors = 0

        for token in tokens:
            token_count += 1
            types.add(token)
            type_count = len(types)
            ttr = type_count / token_count

            if ttr < ttr_threshold:
                factors += 1
                types.clear()
                token_count = 0
                type_count = 0

        # handle the final partial factor
        if token_count > 0:
            partial = (1 - ttr) / (1 - ttr_threshold)
            factors += partial

        return factors

    total_factors = count_factors(tokens)
    return len(tokens) / total_factors if total_factors != 0 else 0.0

def lexical_density(text, lex_pos={"NOUN", "VERB", "ADJ", "ADV"}):
    doc = nlp(text)
    lexical_pos = lex_pos
    total_words = 0
    lexical_words = 0

    for token in doc:
        if token.is_alpha:  # ignore punctuation, numbers, etc.
            total_words += 1
            if token.pos_ in lexical_pos:
                lexical_words += 1

    if total_words == 0:
        return 0.0  # avoid division by zero

    return round((lexical_words / total_words) * 100, 2)

def analyze_text(text):
    if pd.isna(text):
        return pd.Series([None, None, None])
    characters = len(text)
    doc = nlp(text)
    num_tokens = len(doc)
    num_words = sum(1 for token in doc if token.is_alpha)
    num_sentences = len(list(doc.sents))
    mltd_score = ld.mtld([token.lemma_ for token in doc if token.is_alpha])

    lex_dens = lexical_density(text)
    print(f'{text} - {[w for w in doc if w.pos_ in ["NOUN", "VERB", "ADJ", "ADV"] and w.is_alpha]} / {[w for w in doc if w.is_alpha]} = {lex_dens}')
    return pd.Series([lex_dens, mltd_score, characters, num_tokens, num_words, num_sentences])

if __name__ == '__main__':
    target_file_directory = 'C:\\Users\\ASUS\\Documents\\PYTHON_CODE\\Congreso_Louvain_Jul25_01\\sources\\task_prompt_metadata.csv'
    # LOAD CSV FILE TO DATAFRAME
    dataframe = load_file_as_dataframe(target_file_directory)
    dataframe[['Lexical density (content words)', 'MLTD (Measure of Textual Lexical Diversity)', 'characters', 'token', 'words', 'sentences']] = dataframe["Translated task prompt for analysis"].apply(analyze_text)
    dataframe.to_csv(os.path.join('..\\', 'output', 'task_basic_features_ed.csv'))
    df_spacy_analysis = spacy_analysis_to_df(dataframe, "Translated task prompt for analysis")
    df_spacy_analysis.to_csv(os.path.join('..\\', 'output', "spacy_lg_tasks_pos.csv"))
