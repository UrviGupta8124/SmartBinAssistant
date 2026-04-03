import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# 1. Load documents
loader = TextLoader("sample_docs/smart_waste.txt", encoding="utf-8")
documents = loader.load()

# 2. Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# 3. Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="ibm-granite/granite-embedding-30m-english"
)

# 4. Create FAISS vector DB
vector_db = FAISS.from_documents(texts, embedding_model)

# 5. Custom prompt
prompt_template = """You are an AI assistant for the EcoTech Smart Dustbin project by Team Greenovators.

Strict Rules:
- NEVER introduce yourself as a human, person, or use any real person's name
- NEVER say: "it appears", "seems to be", "based on the context", "the text says", "the document mentions", "this website", "I think", "I believe", "it looks like"
- You are an AI assistant, NOT a human team member
- When greeted, say: "Hi! I'm the EcoTech Smart Dustbin AI Assistant. How can I help you?"
- Answer with 100% confidence as a project expert
- Use the conversation history to give connected, context-aware answers

Project Knowledge:
{context}

Question: {question}

Direct Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 6. Load LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 7. QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

def run_query(query: str, chat_history: list = []) -> str:
    # Build history string from last 5 messages
    history_text = ""
    for msg in chat_history[-5:]:
        history_text += f"User: {msg['user']}\nAssistant: {msg['bot']}\n"

    # Inject history into the query
    if history_text:
        full_query = f"Previous conversation:\n{history_text}\nCurrent question: {query}"
    else:
        full_query = query

    response = qa_chain.run(full_query)
    return response