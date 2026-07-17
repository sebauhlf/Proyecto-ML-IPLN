# Proyecto-ML-IPLN
Detección de Clickbait en Titulares de Noticias

Proyecto del curso Introducción al Procesamiento del Lenguaje Natural — Facultad de Ingeniería, UdelaR.

Descripción

Este proyecto aborda un problema de clasificación binaria de texto: determinar si el titular de una noticia es clickbait o no. Se exploran y comparan tres familias de enfoques, desde modelos estadísticos clásicos hasta LLMs modernos usados sin entrenamiento previo (zero-shot / few-shot).

Dataset

Los datos provienen de la competencia internacional TA1C (Te Ahorré Un Click), organizada por el grupo de PLN de FING, cuyo objetivo era impulsar sistemas capaces de detectar y "arruinar" clickbait. Este trabajo se centra en la subtarea de detección (clasificación binaria).


Origen: tesis de maestría de Gabriel Mordecki y su publicación asociada "Te Ahorré Un Click: A Revised Definition of Clickbait and Detection in Spanish News".
Conjuntos: train, dev y test, cada uno con titulares y su anotación (Clickbait / No).
Las clases están desbalanceadas (aprox. 71% "No" / 29% "Clickbait" en dev).


Enfoques implementados

1️⃣ Modelos estadísticos sin embeddings

Clasificadores clásicos entrenados sobre representaciones de texto dispersas:


Vectorización: TF-IDF y Bag of Words (CountVectorizer, n-gramas)
Clasificadores: LogisticRegression, SVC (kernel lineal), GaussianNB, KNeighborsClassifier


2️⃣ Modelos estadísticos con embeddings

Los mismos clasificadores clásicos, pero usando representaciones semánticas densas en lugar de conteos de palabras:


Embeddings: SentenceTransformer (intfloat/multilingual-e5-large y Qwen/Qwen3-Embedding-0.6B)
Clasificadores: LogisticRegression, SVC, GaussianNB, KNeighborsClassifier


3️⃣ Clasificación con LLMs (zero-shot y few-shot)

Uso de modelos de lenguaje generativos, vía prompting, sin fine-tuning:


Modelos: Qwen/Qwen3-4B-Instruct-2507, meta-llama/Llama-3.2-3B-Instruct, ibm-granite/granite-4.0-h-350M
Estrategias de prompting:

Zero-shot, con y sin definición explícita de clickbait en el prompt
Few-shot con ejemplos elegidos al azar
Few-shot con ejemplos seleccionados dinámicamente por similitud semántica (k-NN sobre embeddings del titular a clasificar)





Tecnologías utilizadas


Python
scikit-learn — modelos clásicos (SVM, Regresión Logística, Naive Bayes, KNN), vectorización y métricas
PyTorch — backend de inferencia para los modelos de Hugging Face (manejo de tensores, no_grad, gestión de GPU/CPU)
Hugging Face Transformers — carga y generación con los LLMs (AutoModelForCausalLM, AutoTokenizer, chat templates)
Sentence-Transformers — generación de embeddings de oraciones
Google Colab — entorno de ejecución (GPU)


Evaluación

Todos los enfoques se evalúan con:


F1-score macro (métrica principal, adecuada para el desbalance de clases)
classification_report (precisión, recall y F1 por clase)


Resultados y conclusiones


Los modelos estadísticos (especialmente con embeddings) superaron consistentemente a los LLMs usados vía prompting.
Los LLMs mostraron alta sensibilidad al diseño del prompt: incluir o no una definición explícita de clickbait modificó significativamente el balance entre precisión y recall de la clase minoritaria, en algunos casos de forma contraproducente.
El mejor resultado entre los LLMs se obtuvo con few-shot + selección de ejemplos por k-NN (ibm-granite/granite-4.0-h-350M, F1 macro ≈ 67.3), notablemente por encima de sus variantes zero-shot y few-shot aleatorio.
La brecha de desempeño frente a los modelos estadísticos se atribuye en parte al tamaño reducido de los LLMs usados, en comparación con modelos "estado del arte".


