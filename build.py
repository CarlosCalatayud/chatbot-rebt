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
print("--- INICIANDO BUILD SCRIPT ---")

# Inicializamos Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

# Comprobamos si el índice ya existe en Pinecone
if INDEX_NAME not in pc.list_indexes().names():
    print(f"El índice '{INDEX_NAME}' no existe en Pinecone. Creándolo...")
    # Aquí puedes añadir la lógica para crear el índice si es necesario,
    # pero es mejor crearlo desde la web de Pinecone.
    # pc.create_index(name=INDEX_NAME, dimension=1536, metric='cosine', spec=...)
    raise NameError(f"El índice '{INDEX_NAME}' debe ser creado manualmente en la web de Pinecone primero.")

# Comprobamos si el índice ya ha sido poblado
index_stats = pc.Index(INDEX_NAME).describe_index_stats()
if index_stats.total_vector_count == 0:
    print(f"El índice '{INDEX_NAME}' está vacío. Procesando y subiendo documentos...")
    
    # ... (la carga de PDFs y el text_splitter se mantienen igual) ...
    pdf_files = glob.glob("docs/*.pdf")
    # ...
    all_documents = []
    # ...
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
