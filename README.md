# What Are We Really Asking? A Corpus-Based Study of Task Design in Spanish L2 Writing 
### VAR4LCR Congress - July 2025
This repository contains the code, data, and analysis for the study "What Are We Really Asking? A Corpus-Based Study of Task Design in Spanish L2 Writing", presented at the VAR4LCR Congress on July 7th, 2025. The research explores the linguistic characteristics of writing task prompts across five open-access Spanish L2 learner corpora, with the aim of systematically classifying them based on linguistic complexity indicators. This study combines learner corpus research (LCR) with quantitative, corpus-based methods and unsupervised learning techniques. 
  
## L2 SPANISH CORPORA
We selected five open-access L2 Spanish corpora that include written compositions along with their associated task prompts.
#### Corpus Escrito del Español L2 (CEDEL2) v2.1 (accessed 2025-06-23)
A multi-L1 learner corpus of Spanish as a foreign language that includes spoken and written compositions from learners at various proficiency levels and native control subcorpora. The corpus contains 4,334 compositions, 25 variables and 16 subcorpora—11 learner subcorpora and 5 native control subcorpora. 
#### Corpus de Textos Escritos por Universitarios Taiwaneses Estudiantes de Español

#### Corpus de Textos Escritos por Universitarios Italianos Estudiantes de ELE

#### Leiden Learner Corpus (LLC)

#### Corpus of Written Spanish of L2 and Heritage Speakers (COWS-L2H) (accessed 2025-02-23)

## Objectives
- To analyze the linguistic design of task prompts across five Spanish L2 corpora.
- To identify and classify task types based on linguistic complexity indicators using quantitative methods.

## Research questions
1. What type of task prompts are most frequently used across the five L2 Spanish corpora?
2. What linguistic features characterize the task prompts, and how can they be grouped systematically? 
3. What potential does task prompt design have to reflect variation in communicative register?

## Methodology
1. Data preparation: Extraction of all task prompts and metadata (e.g., task type, modality, time/length/resources limits, corpus population, L1, participant number). Translation of all prompts to Spanish.
2. Linguistic processing: Using spaCy’s es_core_news_lg model for tokenization, lemmatization, POS tagging, morphological analysis, and syntactic parsing.
3. Feature computation: For each task prompt, we calculated linguistic indicators:
Length: number of words, tokens, characters, and sentences.
Lexical complexity: MTLD, lexical density, average word length.
Syntactic complexity: mean sentence length, clause density, subordination ratio, coordinate ratio, complex nominal count, dependency distance, depth of dependency tree.
4. Dimensionality reduction: Principal Component Analysis (PCA) to reduce linguistic features into three composite indicators: length, lexical complexity, syntactic complexity.
5. Clustering:
- Optimal number of clusters determined using the silhouette score.
- K-means clustering applied to group prompts based on the PCA-reduced indicators.
6. Visualization: Normalized PCA scores (range -1 to 1) used to generate radar plots showing average linguistic profiles for each task group.

## Results
1. Cluster distribution
The optimal clustering solution (silhouette method) yielded three distinct task groups.
Group 1 is the most common across all corpora.
Group 0 appears with moderate frequency.
Group 2 is the least represented and consists exclusively of prompts from the Italian corpus.

2. Linguistic profiles
Group 2 exhibits the highest levels of complexity across all three indicators, particularly in syntactic and lexical dimensions.
Group 1 represents intermediate complexity across the board.
Group 0 shows the lowest levels of complexity, making it the simplest task type.
Groups 1 and 2 overlap in their lexical complexity, but differ in length and syntactic indicators.

## Discussion & Conclusion
This study demonstrates that task prompts in L2 Spanish corpora exhibit measurable linguistic variation and can be meaningfully grouped based on their complexity profiles. These findings highlight:
- A lack of standardization in task design across corpora.
- The potential impact of task formulation on learner output, particularly in relation to syntactic and lexical demands.
- The importance of transparency and metadata documentation in corpus-based writing task design.

Future research may explore how these task types correlate with learner performance, or how register and discourse function interact with linguistic complexity in task prompts.

## Repository Structure
sources/    → Original and translated task prompts + metadata + corrected POS labels  
output/     → Processed data, feature counts, PCA results, cluster assignments  
scripts/    → Python code for processing, feature extraction, statistical analysis, visualization  

## CONTACT
- Author: Thuy Huong Nguyen; 
Affiliation: Universidad de Granada;
Email: huong.traductora@gmail.com
