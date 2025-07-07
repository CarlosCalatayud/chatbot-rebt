# build.py

import os
import glob
from dotenv import load_dotenv

# --- Carga de configuración y librerías ---
load_dotenv()
if not all(os.getenv(key) for key in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]):
    raise EnvironmentError("Faltan variables de entorno para el build: OPENAI_API_KEY, PINECONE_API_KEY, o PINECONE_ENVIRONMENT.")

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# --- Constantes ---
INDEX_NAME = "chatbot-rebt" # Asegúrate que coincide con tu índice de Pinecone

# --- Lógica de Creación de la DB ---
# 1. Definir la ruta base del proyecto de forma segura
#    Esto nos da la ruta absoluta de la carpeta donde se encuentra build.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DOCS_PATH = os.path.join(BASE_DIR, 'docs') # Construimos la ruta absoluta a la carpeta 'docs'

print("--- INICIANDO BUILD SCRIPT ---")
print(f"Buscando PDFs en la ruta absoluta: {DOCS_PATH}")


# Inicializamos Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

# Comprobamos si el índice ya existe en Pinecone
if INDEX_NAME not in pc.list_indexes().names():
    raise NameError(f"El índice '{INDEX_NAME}' debe ser creado manualmente en la web de Pinecone primero.")

index_stats = pc.Index(INDEX_NAME).describe_index_stats()

if index_stats.total_vector_count == 0:
    print(f"El índice '{INDEX_NAME}' está vacío. Procesando y subiendo documentos...")
    
    # --- INICIO DE LA SOLUCIÓN: LOGGING DE ARCHIVOS ---
    pdf_files = glob.glob(os.path.join(DOCS_PATH, '*.pdf')) # Usamos la ruta absoluta
    
    if not pdf_files:
        # Añadimos un log para ver qué hay en la carpeta si no encuentra PDFs
        print(f"¡ADVERTENCIA! No se encontraron archivos PDF. Contenido de {DOCS_PATH}: {os.listdir(DOCS_PATH)}")
        raise FileNotFoundError("No se encontraron PDFs en la carpeta 'docs' para el build.")
    
    print(f"Archivos PDF encontrados ({len(pdf_files)}): {pdf_files}")
    # --- FIN DE LA SOLUCIÓN ---
    # ...
    all_documents = []
    # ...
    for pdf_file in pdf_files:
        try:
            print(f"Cargando {pdf_file}...")
            loader = PyPDFLoader(pdf_file)
            all_documents.extend(loader.load())
        except Exception as e:
            print(f"ERROR al cargar el archivo {pdf_file}: {e}")
            # Decidimos si continuar o parar. Por ahora, continuamos.
            continue
    
    print(f"Total de páginas cargadas de todos los PDFs: {len(all_documents)}")
    if not all_documents:
        raise ValueError("No se pudo cargar ninguna página de los PDFs encontrados.")


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # --- INICIO DE LA SOLUCIÓN: SUBIDA POR LOTES ---
    
    # Obtenemos el objeto del índice de Pinecone directamente
    index = pc.Index(INDEX_NAME)
    
    batch_size = 100 # Un tamaño de lote seguro
    print(f"Subiendo {len(texts)} fragmentos a Pinecone en lotes de {batch_size}...")

    for i in range(0, len(texts), batch_size):
        # Seleccionamos un lote de documentos de texto
        batch_of_texts = texts[i:i + batch_size]
        
        # Extraemos solo el contenido de texto para los embeddings
        batch_page_contents = [t.page_content for t in batch_of_texts]
        
        # Extraemos las metadatas
        batch_metadatas = [t.metadata for t in batch_of_texts]
        
        print(f"Procesando lote {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")
        
        # Creamos los embeddings para el lote actual
        embeds = embeddings.embed_documents(batch_page_contents)
        
        # Creamos los IDs únicos para cada vector
        ids = [f"doc_{i+j}" for j in range(len(batch_of_texts))]
        
        # Preparamos los vectores para el upsert
        vectors_to_upsert = zip(ids, embeds, batch_metadatas)
        
        # Hacemos el upsert del lote actual
        index.upsert(vectors=vectors_to_upsert)

    print("¡Documentos subidos a Pinecone con éxito en lotes!")
    # --- FIN DE LA SOLUCIÓN ---
else:
    print(f"El índice '{INDEX_NAME}' ya contiene datos. Build script no necesita hacer nada.")

print("--- BUILD SCRIPT FINALIZADO ---")
