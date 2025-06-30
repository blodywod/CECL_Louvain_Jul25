import pandas as pd
import spacy
import os
from lexical_diversity import lex_div as ld
from lexicalrichness import LexicalRichness
import re
import numpy as np

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

def lexical_density(text, lex_pos={"NOUN", "VERB", "ADJ", "ADV", "AUX"}):
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

def analyze_task(text):
    if pd.isna(text):
        return pd.Series([None, None, None])
    characters = len(text)
    doc = nlp(text)
    num_tokens = len(doc)
    num_words = sum(1 for token in doc if token.is_alpha)
    num_sentences = len(list(doc.sents))
    mltd_score = ld.mtld([token.lemma_ for token in doc if token.is_alpha])

    lex_dens = lexical_density(text)
    # print(f'{text} - {[w for w in doc if w.pos_ in ["NOUN", "VERB", "ADJ", "ADV"] and w.is_alpha]} / {[w for w in doc if w.is_alpha]} = {lex_dens}')
    return pd.Series([lex_dens, mltd_score, characters, num_tokens, num_words, num_sentences])

if __name__ == '__main__':
    task_prompt_directory = '..\\sources\\task_prompt_metadata_01.csv'
    task_pos_directory = '..\\sources\\task_pos_spacy_lg_corrected.csv'
    # LOAD CSV FILE TO DATAFRAME
    df_task_prompt = load_file_as_dataframe(task_prompt_directory)
    df_task_pos = load_file_as_dataframe(task_pos_directory)
    # TASK LINGUISTIC FEATURES ANALYSIS
    df_task_prompt[['Lexical density (content words)', 'MLTD (Measure of Textual Lexical Diversity)', 'characters', 'token', 'words', 'sentences']] = df_task_prompt["Translated task prompt for analysis"].apply(analyze_task)
    df_task_prompt.to_csv(os.path.join('..\\', 'output', 'task_basic_features_ed.csv'))
    df_spacy_analysis = spacy_analysis_to_df(df_task_prompt, "Translated task prompt for analysis")
    df_spacy_analysis.to_csv(os.path.join('..\\', 'output', "task_pos_spacy_lg.csv"))
    # ------------------------------------------------------------------------------------------------------------------
    # CALCULO DEL LEXICAL DENSITY
    # quitamos las puntuaciones
    df_filtered = df_task_pos[df_task_pos['pos_corrected'] != 'PUNCT']
    # agrupamos por prompt y contamos el total de palabras
    word_count = df_filtered.groupby('row_index').size().reset_index(name='word_count')
    # quitamos las palabras que no tienen "significado"
    df_filtered = df_task_pos[df_task_pos['pos_corrected'].isin(["NOUN", "VERB", "ADJ", "ADV", "AUX"])]
    # agrupamos por prompt y contamos el total de palabras con "significado"
    lexical_words_count = df_filtered.groupby('row_index').size().reset_index(name='lexical_words_count')
    # Merge the two counts on row_index
    lexical_df = pd.merge(word_count, lexical_words_count, on='row_index', how='left')
    # Replace NaN lexical counts with 0 (in case some texts have no lexical words)
    lexical_df['lexical_words_count'] = lexical_df['lexical_words_count'].fillna(0)
    # Calculate lexical density
    lexical_df['lexical_density'] = 100 * lexical_df['lexical_words_count'] / lexical_df['word_count']
    # rellenamos la columna de lexical density con los valores calculados
    outputcsv = pd.read_csv(os.path.join('..\\', 'output', 'task_basic_features_ed.csv'))
    outputcsv['Lexical density (content words)'] = lexical_df['lexical_density']
    outputcsv.to_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'))
    # ------------------------------------------------------------------------------------------------------------------
    # # CALCULO DE TTR
    # # quitamos las puntuaciones
    # df_filtered = df_task_pos[df_task_pos['pos_corrected'] != 'PUNCT']
    # # 2. Agrupar por texto y contar tokens (palabras totales)
    # token_count = df_filtered.groupby('row_index').size().reset_index(name='token_count')
    # # 3. Contar tipos únicos por texto usando la columna 'lemma_corrected' o 'token'
    # type_count = df_filtered.groupby('row_index')['token'].nunique().reset_index(name='type_count')
    # # 4. Unir ambas tablas
    # ttr_df = pd.merge(token_count, type_count, on='row_index')
    # # 5. Calcular TTR
    # ttr_df['TTR'] = 100 * ttr_df['type_count'] / ttr_df['token_count']
    # outputcsv = pd.read_csv(os.path.join('..\\', 'output', 'task_basic_features_ed.csv'))
    # outputcsv['TTR'] = ttr_df['TTR']
    # outputcsv.to_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'))
    # ------------------------------------------------------------------------------------------------------------------
    # CALCULO LEXICAL RICHNESS MTLD
    # Filtra solo tokens (sin puntuación)
    df_filtered = df_task_pos[df_task_pos['pos_corrected'] != 'PUNCT']
    # Calcula MTLD para cada grupo por 'row_index'
    mtld_scores = []
    for row_index, group in df_filtered.groupby('row_index'):
        tokens = group['token'].tolist()  # o usa 'lemma_corrected'
        text = ' '.join(tokens)
        lex = LexicalRichness(text)
        mtld = lex.mtld()
        mtld_scores.append({'row_index': row_index, 'MTLD': round(mtld, 2)})
    # Crea un DataFrame con los resultados
    df_mtld = pd.DataFrame(mtld_scores)
    outputcsv = pd.read_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'))
    outputcsv['MTLD'] = df_mtld['MTLD']
    outputcsv.to_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'), index=False)
    # ------------------------------------------------------------------------------------------------------------------
    # CALCULO AVERAGE WORD LENGTH (CHARACTERS)
    # Step 1: Filter out punctuation and space
    df_words = df_task_pos[~df_task_pos['pos_corrected'].isin(['PUNCT', 'SPACE'])].copy()
    # Step 2: Add a new column with word lengths (in characters)
    df_words['word_length'] = df_words['token'].astype(str).str.len()
    # Step 3: Calculate average word length per row_index (prompt)
    avg_word_length = df_words.groupby('row_index')['word_length'].mean().round(2).reset_index(name='average word length (characters)')
    outputcsv = pd.read_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'))
    outputcsv['average word length (characters)'] = avg_word_length['average word length (characters)']
    outputcsv.to_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'), index=False)
    # # ------------------------------------------------------------------------------------------------------------------
    # CALCULO MEAN SENTENCE LENGTH
    # Step 1: Remove punctuation and spaces
    df_words = df_task_pos[~df_task_pos['pos_corrected'].isin(['PUNCT', 'SPACE'])].copy()
    # Step 2: Count number of words per sentence (group by row_index and sent_text)
    sentence_lengths = df_words.groupby(['row_index', 'sent_text']).size().reset_index(name='word_count')
    # Step 3: Calculate average sentence length per prompt (row_index)
    mean_sentence_length = sentence_lengths.groupby('row_index')['word_count'].mean().round(2).reset_index(
        name='mean sentence length')
    # Cargar outputcsv donde guardarás las nuevas columnas
    outputcsv = pd.read_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'))
    outputcsv['mean sentence length'] = mean_sentence_length['mean sentence length']
    outputcsv.to_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'), index=False)
    # ------------------------------------------------------------------------------------------------------------------
    # # CALCULO NOUN CHUNKS
    # '''Hay que volver a calcular este indicador porque la funcion .noun_chunks en Spanish no identifica bien las SN (que, qué, evita)'''
    # df = pd.read_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'))
    # for prompt in df['Translated task prompt for analysis']:
    #     doc = nlp(prompt)
    #     noun_chunks = list(doc.noun_chunks)
    #     num_chunks = len(noun_chunks)
    #     avg_len = round(sum(len(c) for c in noun_chunks) / num_chunks, 2) if num_chunk s > 0 else 0
    # ------------------------------------------------------------------------------------------------------------------
    # CALCULO TOTAL_VERBS (sin AUX_verb), VERB_MOOD_FRQ, DOMINANT_VERB_TENSE, IMPERATIVE_RATIO, SUBJUNCTIVE_RATIO
    # TOTAL_VERBS COUNT
    # Filter only rows where POS is VERB
    df_verbs = df_task_pos[df_task_pos['pos_corrected'] == 'VERB']
    verbs = df_verbs.groupby('row_index').size().reset_index(name='total_verbs')
    outputcsv = pd.read_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'))
    outputcsv = outputcsv.rename(columns={'Unnamed: 0.1': 'row_index'})
    # quitamos la col total_verb en el df para no duplicar
    if 'total_verbs' in outputcsv.columns:
        outputcsv = outputcsv.drop(columns=['total_verbs'])
    # Merge the dataframes:
    outputcsv = outputcsv.merge(verbs[['row_index', 'total_verbs']], on='row_index', how='left')
    outputcsv['total_verbs'] = outputcsv['total_verbs'].fillna(0).astype(int)
    outputcsv.to_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'), index=False)

    # DOMINANT TENSE COUNT
    # Extract mood and tense from 'morph'
    def extract_morph_feature(morph, feature):
        if not isinstance(morph, str):
            return None
        match = re.search(fr'{feature}=([A-Za-z]+)', morph)
        return match.group(1) if match else None

    df_verbs['Tense'] = df_verbs['morphology_corrected'].apply(lambda x: extract_morph_feature(x, 'Tense'))
    # Paso 3: Contar la frecuencia de cada Tense por row_index
    tense_counts = df_verbs.groupby(['row_index', 'Tense']).size().reset_index(name='tense_count')
    # Dominant verb tense per prompt (ignore NaN values)
    dominant_tense = (
        tense_counts.sort_values(['row_index', 'tense_count'], ascending=[True, False])
        .drop_duplicates('row_index', keep='first')[['row_index', 'Tense']]
        .rename(columns={'Tense': 'dominant_tense', 'tense_count': 'dominant_tense_count'})
    )
    outputcsv = pd.read_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'))
    outputcsv = outputcsv.rename(columns={'Unnamed: 0.1': 'row_index'})
    if 'dominant_tense' in outputcsv.columns:
        outputcsv = outputcsv.drop(columns=['dominant_tense'])
    # Paso 7: Hacer merge (left join para mantener todas las filas)
    outputcsv = outputcsv.merge(dominant_tense, on='row_index', how='left')
    # Paso 8: Rellenar con valor por defecto si no hay tense (porque no hay verbos)
    outputcsv['dominant_tense'] = outputcsv['dominant_tense'].fillna('No_verbs')
    outputcsv.to_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'), index=False)

    # IMPERATIVE AND SUBJUNCTIVE RATIO
    df_verbs['Mood'] = df_verbs['morphology_corrected'].apply(lambda x: extract_morph_feature(x, 'Mood'))
    # Paso 4: Contar imperativo y subjuntivo por row_index
    mood_counts = df_verbs.groupby(['row_index', 'Mood']).size().unstack(fill_value=0).reset_index()
    mood_counts['imperative_ratio'] = mood_counts.get('Imp', 0) / (
            mood_counts.get('Imp', 0) + mood_counts.get('Ind', 0) + mood_counts.get('Sub', 0)
    )
    mood_counts['indicative_ratio'] = mood_counts.get('Ind', 0) / (
            mood_counts.get('Imp', 0) + mood_counts.get('Ind', 0) + mood_counts.get('Sub', 0)
    )
    mood_counts['subjunctive_ratio'] = mood_counts.get('Sub', 0) / (
            mood_counts.get('Imp', 0) + mood_counts.get('Ind', 0) + mood_counts.get('Sub', 0)
    )
    # Paso 5: Reemplazar NaN (por división 0/0) con 0
    mood_counts['imperative_ratio'] = mood_counts['imperative_ratio'].fillna(0)
    mood_counts['subjunctive_ratio'] = mood_counts['subjunctive_ratio'].fillna(0)
    mood_counts['indicative_ratio'] = mood_counts['indicative_ratio'].fillna(0)

    # Paso 6: Cargar outputcsv y preparar para merge
    outputcsv = pd.read_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'))
    outputcsv = outputcsv.rename(columns={'Unnamed: 0.1': 'row_index'})
    # Eliminar columnas si ya existen
    for col in ['imperative_ratio', 'subjunctive_ratio', 'indicative_ratio']:
        if col in outputcsv.columns:
            outputcsv = outputcsv.drop(columns=[col])
    # Paso 7: Merge
    outputcsv = outputcsv.merge(mood_counts[['row_index', 'imperative_ratio', 'subjunctive_ratio', 'indicative_ratio']], on='row_index',
                                how='left')
    # Paso 8: Rellenar con 0 para tareas sin verbos
    outputcsv['imperative_ratio'] = outputcsv['imperative_ratio'].fillna(0)
    outputcsv['subjunctive_ratio'] = outputcsv['subjunctive_ratio'].fillna(0)
    outputcsv['indicative_ratio'] = outputcsv['indicative_ratio'].fillna(0)

    # Paso 9: Guardar resultado
    outputcsv.to_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'), index=False)
    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE DEP_DISTANCE AND DEPTH_DEP_TREE
    '''dep_distance: average dependency distance - how far apart words are from their heads, on average
    lower values = simpler, more local constructions (e.g. subject + verb + object)
    High values → more non-local, complex structures (e.g. subordination, embedded clauses).
    depth_dep_tree: maximum depth of the dependency tree in each prompt  - maximum syntactic nesting/how many layers of embedded clauses or modifiers are present. 
    High values → nested, hierarchically deep structures (e.g. relative clauses, multi-level subordination).'''
    # dep_distance calculation
    def compute_dep_distance(text):
        doc = nlp(text)
        distances = [abs(token.i - token.head.i) for token in doc if token.head != token]
        if len(distances) == 0:
            return 0
        return sum(distances) / len(distances)

    # Get a DataFrame with one row per prompt
    df_unique_prompts = df_task_pos[['row_index', 'original_text']].drop_duplicates()
    df_unique_prompts['dep_distance'] = df_unique_prompts['original_text'].apply(compute_dep_distance)

    # depth_dep_tree calculation
    def compute_depth_dep_tree(text):
        def get_depth(token):
            if not list(token.children):  # no children
                return 1
            else:
                return 1 + max(get_depth(child) for child in token.children)

        doc = nlp(text)
        depths = [get_depth(sent.root) for sent in doc.sents]
        if not depths:
            return 0
        return max(depths)

    df_unique_prompts['depth_dep_tree'] = df_unique_prompts['original_text'].apply(compute_depth_dep_tree)

    outputcsv = pd.read_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'))
    for col in ['dep_distance', 'depth_dep_tree']:
        if col in outputcsv.columns:
            outputcsv = outputcsv.drop(columns=[col])
    # Merge back to your main dataframe if needed
    outputcsv = outputcsv.merge(df_unique_prompts[['row_index', 'dep_distance', 'depth_dep_tree']], on='row_index', how='left')
    outputcsv.to_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'), index=False)
    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE CLAUSE DENSITY, COMPLEX NOMINAL MODIFIERS, SUBORDINATION AND COORDINATION INDEX

    # First, group your dep_labels by prompt
    grouped = df_task_pos.groupby('row_index')['dep_labels'].apply(list).reset_index()


    def count_total_clauses(dep_list):
        dep_count = pd.Series(dep_list).value_counts()
        return sum(dep_count.get(dep, 0) for dep in ['ccomp', 'xcomp', 'advcl', 'acl', 'ROOT'])

    def count_subordinations(dep_list):
        dep_count = pd.Series(dep_list).value_counts()
        return sum(dep_count.get(dep, 0) for dep in ['mark', 'advcl', 'ccomp', 'xcomp', 'acl'])

    def count_coordinations(dep_list):
        dep_count = pd.Series(dep_list).value_counts()
        return sum(dep_count.get(dep, 0) for dep in ['cc', 'conj'])

    def count_nominal_mods(dep_list):
        dep_count = pd.Series(dep_list).value_counts()
        return sum(dep_count.get(dep, 0) for dep in ['nmod', 'amod', 'compound', 'appos', 'acl'])

    # Apply function
    grouped['total_clauses'] = grouped.apply(lambda row: count_total_clauses(row['dep_labels']), axis=1)
    grouped['subordinations'] = grouped.apply(lambda row: count_subordinations(row['dep_labels']), axis=1)
    grouped['coordinations'] = grouped.apply(lambda row: count_coordinations(row['dep_labels']), axis=1)
    grouped['nominal_mods'] = grouped.apply(lambda row: count_nominal_mods(row['dep_labels']), axis=1)

    outputcsv = pd.read_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'))
    outputcsv = outputcsv.merge(grouped[
                                    [
                                        'row_index',
                                        'total_clauses',
                                        'subordinations',
                                        'coordinations',
                                        'nominal_mods',
                                        'dep_labels']
                                ], on='row_index', how='left')
    outputcsv.to_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'), index=False)
    outputcsv = pd.read_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'))
    outputcsv['claus_density'] = np.where(
        outputcsv['sentences'] > 0,
        outputcsv['total_clauses'] / outputcsv['sentences'],
        0
    )
    outputcsv['sub_ratio'] = np.where(
        outputcsv['total_clauses'] > 0,
        outputcsv['subordinations'] / outputcsv['total_clauses'],
        0
    )
    outputcsv['coord_ratio'] = np.where(
        outputcsv['total_clauses'] > 0,
        outputcsv['coordinations'] / outputcsv['total_clauses'],
        0
    )
    outputcsv['nominal_ratio'] = np.where(
        outputcsv['total_clauses'] > 0,
        outputcsv['nominal_mods'] / outputcsv['total_clauses'],
        0
    )
    outputcsv.to_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'), index=False)

    # ------------------------------------------------------------------------------------------------------------------
    # TASK PROMPT CLASSIFICATION - PCA & K-MEANS
    from sklearn.decomposition import PCA

    # LENGTH
    length_features = ['token', 'characters', 'words', 'sentences']
    pca_length = PCA(n_components=1)
    outputcsv['length_indicator'] = pca_length.fit_transform(outputcsv[length_features])
    # LEXICAL
    lexical_features = ['MTLD', 'Lexical density (content words)', 'average word length (characters)']
    pca_lexical = PCA(n_components=1)
    outputcsv['lexical_indicator'] = pca_lexical.fit_transform(outputcsv[lexical_features])
    # SYNTACTIC
    syntactic_features = [
        'mean sentence length',
        'imperative_ratio',
        'subjunctive_ratio',
        'indicative_ratio',
        'dep_distance',
        'depth_dep_tree',
        'claus_density',
        'sub_ratio',
        'coord_ratio',
        'nominal_ratio'
    ]
    pca_syntactic = PCA(n_components=1)
    outputcsv['syntactic_indicator'] = pca_syntactic.fit_transform(outputcsv[syntactic_features])

    # CLUSTERING ACCORDING TO 3 INDICATORS (LENGTH, LEXICAL COMPLEX, SINTACTIC COMPLEX
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt

    # 1. Define the features to use for clustering (your PCA indicators)
    features_for_clustering = outputcsv[['length_indicator', 'lexical_indicator', 'syntactic_indicator']]
    # 2. Range of k to test
    K = range(2, 11)  # k must be >= 2 for silhouette
    silhouette_scores = []

    # 3. Loop through different k values
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features_for_clustering)
        score = silhouette_score(features_for_clustering, labels)
        silhouette_scores.append(score)
        print(f"Silhouette score for k={k}: {score:.4f}")

    # 4. Plot the silhouette scores
    plt.figure(figsize=(8, 4))
    plt.plot(K, silhouette_scores, 'ro-', linewidth=2)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score by Number of Clusters')
    plt.grid(True)
    plt.show()

    kmeans = KMeans(n_clusters=3, random_state=42)  # or choose optimal k using elbow method
    outputcsv['task_group'] = kmeans.fit_predict(features_for_clustering)
    outputcsv.to_csv(os.path.join('..\\', 'output', 'task_basic_features_ed_01.csv'), index=False)

