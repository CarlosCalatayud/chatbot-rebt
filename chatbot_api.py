# chatbot_api.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# --- 1. CARGA DE CONFIGURACIÓN Y LIBRERÍAS ---
load_dotenv()
# Verificamos las variables de entorno necesarias para EJECUTAR la API
if not all(os.getenv(key) for key in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]):
    raise EnvironmentError("Faltan variables de entorno para ejecutar la API.")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore

# --- 2. CONFIGURACIÓN DE CONSTANTES Y CADENA RAG ---
INDEX_NAME = "rebt-chatbot"

print("Iniciando Bot de LangChain (modo de ejecución)...")
print("Configurando la cadena RAG con Pinecone...")

# Cargamos el modelo de lenguaje
llm = ChatOpenAI(model="gpt-4o")

# Cargamos los embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Cargamos nuestro vector store directamente desde el índice de Pinecone ya existente
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME, 
    embedding=embeddings
)

# Creamos el retriever, que buscará en Pinecone
retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

# Definimos el prompt del sistema (sin cambios)
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

# Creamos las cadenas de LangChain (sin cambios)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("¡Cadena RAG lista para recibir preguntas!")

# --- 3. CREACIÓN DE LA API CON FLASK ---
app = Flask(__name__)
CORS(app) # Permitimos peticiones desde Lovable

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"error": "Petición incorrecta. Se requiere un campo 'question'."}), 400

    user_question = data['question']
    print(f"API recibió la pregunta: {user_question}")

    try:
        response = rag_chain.invoke({"input": user_question})
        
        disclaimer = (
            "\n\n---\n"
            "*Aviso: Esta respuesta es generada por IA y debe ser usada como una guía. "
            "Verifica siempre la información con el texto oficial del REBT.*"
        )
        
        final_answer = response.get("answer", "No se pudo generar una respuesta.") + disclaimer
        
        return jsonify({"answer": final_answer})

    except Exception as e:
        print(f"Error al procesar la pregunta: {e}")
        return jsonify({"error": "Ocurrió un error interno al procesar tu pregunta."}), 500

# Punto de entrada para correr el servidor Flask (para pruebas locales)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)