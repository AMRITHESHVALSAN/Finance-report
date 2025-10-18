import streamlit as st
import os
import pandas as pd
import numpy as np
import faiss
# REMOVED: from dotenv import load_dotenv - no longer needed
from google import generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModel, AutoTokenizer
import torch

# --- 1. SETUP AND CONFIGURATION ---

st.set_page_config(page_title="Financial Report AI", layout="wide")

# --- MODIFICATION FOR STREAMLIT DEPLOYMENT ---
# The app will now get the key from Streamlit's Secrets Manager
# REMOVED: load_dotenv()
# API_KEY = os.getenv("GOOGLE_API_KEY") 
# ADDED:
API_KEY = st.secrets["GOOGLE_API_KEY"]
# -----------------------------------------------

if not API_KEY:
    st.error("Google API key not found in Streamlit Secrets. Please add it.")
    st.stop()

try:
    genai.configure(api_key=API_KEY)
    llm = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"Failed to configure Gemini: {e}")
    st.stop()

# --- 2. CACHED FUNCTIONS (No changes here) ---

@st.cache_resource
def load_embedding_model():
    """Loads the sentence-transformer model and tokenizer, caches them."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        return tokenizer, model
    except Exception as e:
        st.error(f"Could not load embedding model: {e}")
        st.stop()

@st.cache_data
def generate_initial_report(file_path):
    """Reads financial data and generates a management report."""
    def excel_multiple_sheets_to_string(path: str) -> str:
        try:
            all_sheets_df = pd.read_excel(path, sheet_name=None)
            combined_data_string = ""
            for sheet_name, df in all_sheets_df.items():
                combined_data_string += f"--- START OF SHEET: {sheet_name} ---\n"
                combined_data_string += df.to_csv(index=False)
                combined_data_string += f"--- END OF SHEET: {sheet_name} ---\n\n"
            return combined_data_string
        except FileNotFoundError:
            st.error(f"Error: The file '{path}' was not found.")
            st.stop()
        except Exception as e:
            st.error(f"An error occurred while reading the Excel file: {e}")
            st.stop()

    data_string = excel_multiple_sheets_to_string(file_path)
    mgmnt_prompt = f'''You are a seasoned financial analyst. Your task is to generate a concise, insightful, and professional quarterly management report. The report must be based exclusively on the provided Balance Sheet (BS), Income Statement (IS), and Equity Statement.

    Your report must include the following sections and analytical content:

    1.  **Executive Summary:**
        * Provide a high-level summary of the company's financial health and performance for the quarter.
        * Identify the most significant insights from the Income Statement and Balance Sheet and state them clearly.

    2.  **Financial Performance Analysis (Based on the Income Statement):**
        * Report the **Total Revenue**, **Gross Profit**, and **Net Income** for the quarter.
        * Calculate and state the **Gross Profit Margin** (Gross Profit / Total Revenue) as a percentage. Explain what this number reveals about the company's operational efficiency.
        * Calculate and state the **Net Profit Margin** (Net Income / Total Revenue) as a percentage. Explain what this number indicates about the company's overall profitability.

    3.  **Financial Position Analysis (Based on the Balance Sheet):**
        * Report the **Total Assets**, **Total Liabilities**, and **Total Equity**.
        * Calculate and state the **Current Ratio** (Current Assets / Current Liabilities). Interpret this number to explain the company's short-term liquidity.
        * Calculate and state the **Debt-to-Equity Ratio** (Total Liabilities / Total Equity). Interpret this number to assess the company's leverage and financial risk.

    4.  **Key Insights and Recommendations:**
        * Synthesize the findings from all your calculations and analysis.
        * Provide a maximum of three key insights about the company's performance or position.
        * Based on these insights, provide one actionable recommendation for management to consider.

    Use the provided data exclusively to derive your reports. 
    {data_string}
    The final report must be a single, well-structured text document. Do not include raw financial data in the final report.'''

    try:
        response = llm.generate_content(mgmnt_prompt)
        return response.text
    except Exception as e:
        st.error(f"Failed to generate report with Gemini: {e}")
        st.stop()

@st.cache_resource
def build_rag_system(_report_text, _tokenizer, _model):
    """Chunks text, generates embeddings, and builds a FAISS index."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
    chunks = text_splitter.split_text(_report_text)

    def get_embeddings_in_batch(texts, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = _tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = _model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            all_embeddings.extend(batch_embeddings)
        return np.array(all_embeddings).astype('float32')

    embeddings = get_embeddings_in_batch(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, chunks

# --- 3. RAG QUERY FUNCTIONS (No changes here) ---

def get_rag_answer(query, index, chunks, tokenizer, model):
    """Performs the full RAG process to answer a query."""
    def get_query_embedding(text, tokenizer, model):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state
        return torch.mean(outputs, dim=1).numpy().astype('float32')

    query_embedding = get_query_embedding(query, tokenizer, model)
    k = 3
    distances, indices = index.search(query_embedding, k)
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    context_string = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""
    You are a highly skilled financial analyst. Your task is to analyze the provided context and provide a comprehensive and insightful answer to the user's question.

    Your response must:
    1.  **Synthesize information** from all provided chunks to create a coherent and complete answer. Do not just list facts.
    2.  **Strictly use only the information provided** in the context. Do not use any external knowledge.
    3.  Use the data as context and your own knowledge and capabilities generate a meaningful answer for the user query.
    4.  If the context does not contain the information needed to answer, state that the information is not available.

    User Query: {query}

    Context:
    ---
    {context_string}
    ---

    Answer:"""
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {e}"

# --- 4. STREAMLIT UI AND APPLICATION FLOW (No changes here) ---

st.title("📄 AI-Powered Financial Report Analyzer")
st.markdown("Generate a C-suite level financial report from your data and then ask specific questions about it.")

if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
    st.session_state.report_text = ""
    st.session_state.faiss_index = None
    st.session_state.chunks = []
    st.session_state.messages = []

tokenizer, embedding_model = load_embedding_model()

if not st.session_state.report_generated:
    st.subheader("Generate Management Report")
    if st.button("Generate Financial Report from Excel"):
        with st.spinner('Generating financial report... This may take a moment.'):
            report_text = generate_initial_report('ABC_Financials_12312024.xlsx')
            st.session_state.report_text = report_text

        with st.spinner('Creating searchable index from the report...'):
            faiss_index, chunks = build_rag_system(report_text, tokenizer, embedding_model)
            st.session_state.faiss_index = faiss_index
            st.session_state.chunks = chunks

        st.session_state.report_generated = True
        st.rerun()

else:
    st.success("Report generated and ready for analysis!")

    with st.expander("📂 View the Full Generated Financial Report"):
        st.markdown(st.session_state.report_text)

    st.header("💬 Chat with Your Report")
    st.markdown("Ask a question, and the AI will answer based on the document's content.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("e.g., What is the debt-to-equity ratio?"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Analyzing report to find an answer..."):
            assistant_response = get_rag_answer(
                user_query,
                st.session_state.faiss_index,
                st.session_state.chunks,
                tokenizer,
                embedding_model
            )
        
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})