# chatbot_api.py
import os
import glob
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# --- 1. CARGA DE CONFIGURACIÓN Y LIBRERÍAS (igual que antes) ---
load_dotenv()
if not all(os.getenv(key) for key in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]):
    raise EnvironmentError("Faltan variables de entorno: OPENAI_API_KEY, PINECONE_API_KEY, o PINECONE_ENVIRONMENT.")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Importaciones de Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# --- 2. CONFIGURACIÓN DE PINECONE Y LANGCHAIN ---
INDEX_NAME = "chatbot-rebt" # El nombre de tu índice en Pinecone

# Inicializamos Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

# --- 3. PROCESAMIENTO Y CARGA DE LA BASE DE DATOS VECTORIAL ---
# Esta parte se ejecuta UNA SOLA VEZ cuando el servidor arranca.
print("Iniciando Bot de LangChain...")

# Comprobamos si el índice ya ha sido poblado
# (Una forma simple es verificar si el índice en Pinecone tiene vectores)
index_stats = pc.Index(INDEX_NAME).describe_index_stats()
if index_stats.total_vector_count == 0:
    print(f"El índice '{INDEX_NAME}' está vacío. Procesando y subiendo documentos...")
    
    # Cargar y procesar los PDFs (esto solo se ejecutará la primera vez que arranque el servidor)
    import glob
    pdf_files = glob.glob("docs/*.pdf")
    if not pdf_files:
        raise FileNotFoundError("No se encontraron PDFs en la carpeta 'docs'.")
        
    all_documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        all_documents.extend(loader.load())
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    
    # Subir los documentos a Pinecone
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=INDEX_NAME)
    print("¡Documentos subidos a Pinecone con éxito!")
else:
    print(f"El índice '{INDEX_NAME}' ya contiene datos. Cargando vector store existente.")


# --- 4. CONFIGURACIÓN DE LA CADENA RAG (usando Pinecone) ---
print("Configurando la cadena RAG con Pinecone...")
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Cargamos el vector store desde Pinecone
vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 7})


if not os.path.exists(PERSIST_DIRECTORY):
    print(f"Base de datos vectorial en '{PERSIST_DIRECTORY}' no encontrada. Creando una nueva...")
    pdf_files = glob.glob("docs/*.pdf")
    if not pdf_files:
        raise FileNotFoundError("Error: No se encontraron archivos PDF en la carpeta 'docs'.")
    
    all_documents = []
    for pdf_file in pdf_files:
        print(f"Cargando {pdf_file}...")
        loader = PyPDFLoader(pdf_file)
        all_documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print(f"Creando y guardando la base de datos vectorial en '{PERSIST_DIRECTORY}'...")
    Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)
    print("¡Base de datos creada y guardada con éxito!")
else:
    print(f"Base de datos vectorial encontrada en '{PERSIST_DIRECTORY}'.")

# --- 4. CONFIGURACIÓN DE LA CADENA RAG (se carga en memoria) ---
print("Configurando la cadena RAG...")
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

system_prompt = (
    "Eres un asistente experto y un formador técnico para electricistas profesionales en España. Tu conocimiento se basa exclusivamente en el REBT y sus Guías Técnicas de Aplicación. "
    "Actúa como un experto seguro de sí mismo. Tus respuestas deben ser directas, precisas y estructuradas."
    "1. Primero, responde directamente a la pregunta del usuario. "
    "2. Después, justifica tu respuesta basándote en el contexto proporcionado. "
    "3. **Cita siempre la ITC-BT o Guía Técnica específica** de la que extraes la información. Esto es obligatorio. "
    "4. Si el contexto no contiene la respuesta, di claramente: 'Basado en la documentación disponible, no tengo una respuesta precisa para tu consulta.' No inventes información."
    "Utiliza negritas para destacar los conceptos y normativas clave."
    "\n\n"
    "Contexto: {context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
print("¡Cadena RAG lista para recibir preguntas!")

# --- 5. CREACIÓN DE LA API CON FLASK ---
app = Flask(__name__)
CORS(app) # Permitimos peticiones desde cualquier origen (Lovable)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"error": "Petición incorrecta. Se requiere un campo 'question'."}), 400

    user_question = data['question']
    print(f"API recibió la pregunta: {user_question}")

    try:
        # Invocamos la cadena RAG con la pregunta
        response = rag_chain.invoke({"input": user_question})
        
        disclaimer = (
            "\n\n---\n"
            "*Aviso: Esta respuesta es generada por IA y debe ser usada como una guía. "
            "Verifica siempre la información con el texto oficial del REBT.*"
        )
        
        final_answer = response.get("answer", "No se pudo generar una respuesta.") + disclaimer
        
        # Devolvemos una respuesta JSON que Lovable pueda entender
        return jsonify({"answer": final_answer})

    except Exception as e:
        print(f"Error al procesar la pregunta: {e}")
        return jsonify({"error": "Ocurrió un error interno al procesar tu pregunta."}), 500

# Punto de entrada para correr el servidor Flask
if __name__ == '__main__':
    # Render usará gunicorn, esto es para pruebas locales
    app.run(host='0.0.0.0', port=5002, debug=True)