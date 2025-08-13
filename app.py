# app.py
# The Streamlit application for OrchestraAI, including V2 Figma Draft Generation.

import os
import numpy as np
import pandas as pd
from pyairtable import Api
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import streamlit as st
import streamlit.components.v1 as components
import time

# --- 1. CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="OrchestraAI", layout="wide")

st.title("🚀 OrchestraAI Designer's Copilot")
st.markdown("This tool analyzes a new client brief, retrieves relevant context from our Airtable knowledge base, and prepares a comprehensive Kick-off Packet.")

# Use Streamlit secrets for robust key management
# In a real deployed app, you would set these in Streamlit's secret management
AIRTABLE_API_KEY = "patsNSk4hUHOBMFlM.bf2a4a188bc372286b5196272e394bf4eba94aa0c3e8acb79a8742efb8a34404"
AIRTABLE_BASE_ID = "appU2kDCpzCTLxrNp"
GEMINI_API_KEY = "AIzaSyAW1zwSs79MA7Uv6-AgC2lLyHuzBjeI8Zs"

DELIVERABLES_TABLE_NAME = "Deliverables"
INSPIRATIONS_TABLE_NAME = "Inspirations"

# --- 2. SETUP THE AI AND DATA CONNECTIONS (CACHED) ---
# Use Streamlit's caching to load models and data only once.
@st.cache_resource
def get_ai_models():
    print("Loading AI models...")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return gemini_model, embedding_model

@st.cache_data
def get_airtable_data():
    print("Fetching and caching Airtable data...")
    airtable_api = Api(AIRTABLE_API_KEY)
    
    # Fetch Deliverables
    deliverables_table = airtable_api.table(AIRTABLE_BASE_ID, DELIVERABLES_TABLE_NAME)
    all_deliverables = deliverables_table.all()
    deliverables_df = pd.DataFrame([record['fields'] for record in all_deliverables])
    
    # --- DATA CLEANING STEP ---
    deliverables_df.dropna(how='all', inplace=True) # Remove empty rows
    deliverables_df.fillna('', inplace=True) # Replace NaN with empty strings
    
    deliverables_df['search_text'] = deliverables_df['Orchestra task name'] + ". " + deliverables_df['Notes on Timing (if Late/Early)']

    # Fetch Inspirations
    inspirations_table = airtable_api.table(AIRTABLE_BASE_ID, INSPIRATIONS_TABLE_NAME)
    all_inspirations = inspirations_table.all()
    inspirations_df = pd.DataFrame([record['fields'] for record in all_inspirations])

    # --- DATA CLEANING STEP ---
    inspirations_df.dropna(how='all', inplace=True) # Remove empty rows
    inspirations_df.fillna('', inplace=True) # Replace NaN with empty strings

    inspirations_df['search_text'] = inspirations_df['Nom'] + ". Tags: " + inspirations_df['Type']
    
    return deliverables_df, inspirations_df

gemini_model, embedding_model = get_ai_models()
deliverables_df, inspirations_df = get_airtable_data()

# --- 3. CORE LOGIC ---
@st.cache_data
def get_embeddings(_df, column_name):
    print(f"Generating embeddings for {column_name}...")
    return embedding_model.encode(_df[column_name].tolist())

deliverable_embeddings = get_embeddings(deliverables_df, 'search_text')
inspiration_embeddings = get_embeddings(inspirations_df, 'search_text')

def find_similar_items(query, df, embeddings, top_k=3):
    query_embedding = embedding_model.encode([query])
    cos_scores = np.dot(embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(cos_scores)[-top_k:][::-1]
    return df.iloc[top_indices]

def generate_kickoff_packet_from_brief(client_brief):
    deliverable_cols_to_show = ['Orchestra task name', 'Designer Name', 'Link to Deliverable (Figma ou Frame.io)']
    inspiration_cols_to_show = ['Nom', 'Lien', 'Type']

    similar_deliverables = find_similar_items(client_brief, deliverables_df, deliverable_embeddings, top_k=3)
    similar_inspirations = find_similar_items(client_brief, inspirations_df, inspiration_embeddings, top_k=5)
    
    prompt = f"""
    You are "OrchestraAI," an expert AI assistant. Your role is to analyze a new client brief and our internal data to prepare a "Project Kick-off Packet".

    **New Client Brief:**
    "{client_brief}"

    ---
    **Context from our Airtable - Top 3 Similar Past Deliverables:**
    {similar_deliverables[deliverable_cols_to_show].to_markdown(index=False)}
    ---
    **Context from our Airtable - Top 5 Relevant Visual Inspirations:**
    {similar_inspirations[inspiration_cols_to_show].to_markdown(index=False)}
    ---

    **Your Task:**
    Based on the new brief and ALL the provided context, generate a complete and helpful "Project Kick-off Packet" in Markdown format. The packet should include:
    1.  A brief analysis of the new project.
    2.  A section called "## Recommended Past Projects" containing the table of deliverables from the context.
    3.  A section called "## Visual Inspiration" containing the table of inspirations from the context.
    4.  A "## Brief Completeness Check" section with clarifying questions for the client.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: Could not generate response from Gemini AI. Details: {e}"

def generate_figma_draft_html(kickoff_packet):
    """ (V2 FEATURE) Takes the generated packet and creates an HTML draft. """
    prompt = f"""
    You are an expert front-end developer who specializes in creating simple, clean prototypes with HTML and Tailwind CSS.
    Based on the following "Project Kick-off Packet", generate a single HTML file that represents a first draft of the described webpage.

    **CRITICAL INSTRUCTIONS:**
    - Use Tailwind CSS for all styling. Load it from the CDN: `<script src="https://cdn.tailwindcss.com"></script>`.
    - Use appropriate semantic HTML5 tags (`<header>`, `<nav>`, `<main>`, `<section>`, `<footer>`).
    - Use placeholder images from `https://placehold.co/` where necessary (e.g., `https://placehold.co/600x400`).
    - The design should be clean, modern, and directly reflect the analysis and goals described in the packet.
    - The output must be ONLY the HTML code, starting with `<!DOCTYPE html>` and ending with `</html>`. Do not include any explanations or surrounding text.

    ---
    **Project Kick-off Packet:**
    {kickoff_packet}
    ---
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        # Clean up the response to ensure it's just HTML
        html_code = response.text.strip()
        if html_code.startswith("```html"):
            html_code = html_code[7:]
        if html_code.endswith("```"):
            html_code = html_code[:-3]
        return html_code
    except Exception as e:
        return f"Error generating HTML draft: {e}"

# --- 4. STREAMLIT UI ---
client_brief = st.text_area("Enter Client Brief:", height=150, placeholder="e.g., We need a new e-commerce website for our coffee brand...")

if st.button("Generate Kick-off Packet", type="primary"):
    if client_brief:
        with st.spinner("🧠 Analyzing brief and searching knowledge base..."):
            kickoff_packet = generate_kickoff_packet_from_brief(client_brief)
        
        st.divider()
        st.header("📄 Project Kick-off Packet")
        st.markdown(kickoff_packet)
        
        with st.spinner("🎨 Generating V2 Figma Draft (HTML)..."):
            time.sleep(1) # Small delay for better UX
            figma_html = generate_figma_draft_html(kickoff_packet)
        
        st.divider()
        st.header("✨ V2: Figma Draft (HTML)")
        st.markdown("Use a Figma plugin like [HTML to Design](https://www.figma.com/community/plugin/1159123024925448726/html-to-design) to import this code and create a live draft.")
        
        # --- NEW TABBED VIEW ---
        tab1, tab2 = st.tabs(["📄 Code", "🖼️ Result"])

        with tab1:
            st.code(figma_html, language="html")
        
        with tab2:
            components.html(figma_html, height=600, scrolling=True)

    else:
        st.warning("Please enter a client brief.")
