# What Are We Really Asking? A Corpus-Based Study of Task Design in Spanish L2 Writing 
### VAR4LCR Conference - July 2025
### Project summary
This repository contains the code, data, and analysis for the study "What Are We Really Asking? A Corpus-Based Study of Task Design in Spanish L2 Writing", presented at the VAR4LCR Congress on July 7th, 2025. The study investigates the linguistic design of writing task prompts in five open-access Spanish L2 learner corpora. By applying quantitative corpus-based methods and unsupervised learning (PCA and clustering), it systematically classifies prompts according to linguistic complexity. Bridging learner corpus research and NLP, the study highlights that even the prompts themselves shape learner performance through their linguistic demands. By making these task characteristics visible, we move toward more transparent and standardized task design—laying the groundwork for more valid, interpretable, and comparable learner data in L2 research.
  
## L2 SPANISH CORPORA
I selected five open-access written L2 Spanish corpora that include written compositions along with their associated task prompts.
#### Corpus Escrito del Español L2 (CEDEL2) v2.1 (accessed 2025-06-23)
A multi-L1 learner corpus of Spanish as a foreign language that includes spoken and written compositions from learners at various proficiency levels and native control subcorpora. The corpus contains 4,334 compositions, 25 variables and 16 subcorpora—11 learner subcorpora and 5 native control subcorpora. 
#### Corpus de Textos Escritos por Universitarios Taiwaneses Estudiantes de Español

#### Corpus de Textos Escritos por Universitarios Italianos Estudiantes de ELE

#### Leiden Learner Corpus (LLC)

#### Corpus of Written Spanish of L2 and Heritage Speakers (COWS-L2H) (accessed 2025-02-23)

## Objectives
- To analyze the linguistic design of task prompts across five L2 Spanish corpora.
- To identify and classify task types based on linguistic complexity indicators using quantitative methods.

## Research questions
RQ1: How can we systematically classify/group task prompts by linguistic complexity?
RQ2: What types of task prompts are most frequently used across the five L2 Spanish corpora? 
RQ3: How does the linguistic complexity of task prompts vary across different task groups?

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
#### RQ1: How can we systematically classify/group task prompts by linguistic complexity?
The optimal clustering solution (silhouette method) yielded three distinct task groups.
•	Group 2 = Highest complexity: longer prompts, more clauses, higher lexical diversity.
•	Group 0 = Intermediate: moderate length, average syntactic depth, some lexical richness.
•	Group 1 = Lowest: short prompts, simple syntax, limited vocabulary.

#### RQ2: What types of task prompts are most frequently used across the five L2 Spanish corpora? 
Task group 1 = 61% the most common across all corpora.
Task group 2 = 8.4% the least represented and consists exclusively of prompts from the Italian corpus.
Task group 0 = 30.6% (LLC & CEDEL2) appears with moderate frequency.

#### RQ3: How does the linguistic complexity of task prompts vary across different task groups? (a demo, work-in-progress)
By analyzing the mean and standard deviation of key features, I found:
•	Lexical complexity: All groups showed similar lexical density and average word length, but Group 0 and Group 2 had much higher MTLD values than Group 1, indicating greater lexical diversity. However, Group 0 displayed the highest variability, suggesting inconsistent task prompt across proficiency levels.
•	Syntactic complexity: Group 2 consistently outperformed the others in clause density, subordination, and dependency distance, with low variability, indicating uniform use of complex structures. In contrast, Group 1 showed the simplest syntax, while Group 0 showed moderate complexity but high variability, pointing to uneven task engagement or learner proficiency.
These results validate the clustering and provide insight into the linguistic demands of each task group.

## Conclusions and Future directions
#### Main Findings
•	Task prompts can be meaningfully grouped into high, intermediate, and low linguistic complexity levels.
•	The simplest tasks (Group 1) are also the most used in L2 Spanish corpus compilation.
•	Groups 0 and 2 show richer vocabulary and more complex syntax.
•	Group 0 is more variable, likely reflecting tasks targeted to specific proficiency levels.

#### Limitations
•	There’s a lack of standardization in task design across corpora.
•	This highlights the importance of metadata transparency for future corpus-based studies.
•	The sample size is relatively limited, so more data are needed to generalize the results.
•	The spaCy lg model is efficient but not always accurate, so some errors in tagging persist.

#### Future Directions
•	Link these task types to actual learner written performance to explore how complexity affects output.
•	Conduct qualitative analyses of individual prompts (e.g., lexical choices, sentence structures) to complement PCA findings.
•	Expand the dataset with more corpora and task types for greater generalizability.
•	Investigate how register or genre (e.g., informal letters, academic writing) relates to linguistic complexity in prompts.
•	Test more advanced NLP tools, like spaCy’s transformer model, for improved annotation accuracy.


## Repository Structure
sources/    → Original and translated task prompts + metadata + corrected POS labels  
output/     → Processed data, feature counts, PCA results, cluster assignments  
scripts/    → Python code for processing, feature extraction, statistical analysis, visualization  

## CONTACT
- Author PhD candidate: Thuy Huong Nguyen; 
Affiliation: Universidad de Granada;
Email: huong.traductora@gmail.com
