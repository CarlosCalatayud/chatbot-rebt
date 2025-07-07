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
INDEX_NAME = "rebt-chatbot" # Asegúrate que coincide con tu índice de Pinecone

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
    
    pdf_files = glob.glob("docs/*.pdf")
    if not pdf_files:
        raise FileNotFoundError("No se encontraron PDFs en la carpeta 'docs' para el build.")
        
    all_documents = []
    for pdf_file in pdf_files:
        print(f"Cargando {pdf_file}...")
        loader = PyPDFLoader(pdf_file)
        all_documents.extend(loader.load())
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Subimos los documentos a Pinecone
    print(f"Subiendo {len(texts)} fragmentos a Pinecone...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=INDEX_NAME)
    print("¡Documentos subidos a Pinecone con éxito!")
else:
    print(f"El índice '{INDEX_NAME}' ya contiene datos. Build script no necesita hacer nada.")

print("--- BUILD SCRIPT FINALIZADO ---")