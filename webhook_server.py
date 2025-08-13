# webhook_server.py
# An automated AI agent triggered by webhooks.

import os
import numpy as np
import pandas as pd
from pyairtable import Api
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from flask import Flask, request, jsonify
from datetime import datetime

# --- 1. CONFIGURATION ---
# It's good practice to load these from environment variables rather than hardcoding.
AIRTABLE_API_KEY = "patsNSk4hUHOBMFlM.bf2a4a188bc372286b5196272e394bf4eba94aa0c3e8acb79a8742efb8a34404"
AIRTABLE_BASE_ID = "appU2kDCpzCTLxrNp"
DELIVERABLES_TABLE_NAME = "Deliverables"
INSPIRATIONS_TABLE_NAME = "Inspirations"
GEMINI_API_KEY = "AIzaSyAW1zwSs79MA7Uv6-AgC2lLyHuzBjeI8Zs"

# --- 2. SETUP THE AI AND DATA CONNECTIONS ---
print("🚀 Initializing OrchestraAI Webhook Server...")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
airtable_api = Api(AIRTABLE_API_KEY)
deliverables_table = airtable_api.table(AIRTABLE_BASE_ID, DELIVERABLES_TABLE_NAME)
inspirations_table = airtable_api.table(AIRTABLE_BASE_ID, INSPIRATIONS_TABLE_NAME)
print("   -> Caching Airtable data for fast responses...")

# --- 3. REUSABLE CORE LOGIC ---

def get_and_embed_data():
    """
    Fetches data from Airtable (or loads from cache) and generates embeddings.
    This version is more robust against missing columns.
    """
    # Load Deliverables Data
    if os.path.exists('deliverables.pkl') and os.path.exists('deliverable_embeddings.npy'):
        deliverables_df = pd.read_pickle('deliverables.pkl')
        deliverable_embeddings = np.load('deliverable_embeddings.npy')
    else:
        all_deliverables = deliverables_table.all()
        deliverables_df = pd.DataFrame([record['fields'] for record in all_deliverables])
        # Robustly handle potentially missing column before creating search_text
        if 'Notes on Timing (if Late/Early)' not in deliverables_df.columns:
            deliverables_df['Notes on Timing (if Late/Early)'] = ''
        deliverables_df['search_text'] = deliverables_df['Orchestra task name'].fillna('') + ". " + deliverables_df['Notes on Timing (if Late/Early)'].fillna('')
        deliverable_embeddings = embedding_model.encode(deliverables_df['search_text'].tolist())
        deliverables_df.to_pickle('deliverables.pkl')
        np.save('deliverable_embeddings.npy', deliverable_embeddings)

    # Load Inspirations Data
    if os.path.exists('inspirations.pkl') and os.path.exists('inspiration_embeddings.npy'):
        inspirations_df = pd.read_pickle('inspirations.pkl')
        inspiration_embeddings = np.load('inspiration_embeddings.npy')
    else:
        all_inspirations = inspirations_table.all()
        inspirations_df = pd.DataFrame([record['fields'] for record in all_inspirations])
        inspirations_df['search_text'] = inspirations_df['Nom'].fillna('') + ". Tags: " + inspirations_df['Type'].fillna('')
        inspiration_embeddings = embedding_model.encode(inspirations_df['search_text'].tolist())
        inspirations_df.to_pickle('inspirations.pkl')
        np.save('inspiration_embeddings.npy', inspiration_embeddings)
        
    return deliverables_df, deliverable_embeddings, inspirations_df, inspiration_embeddings

def find_similar_items(query, df, embeddings, top_k=3):
    """Finds the most similar items in a dataframe to a query string."""
    query_embedding = embedding_model.encode([query])
    cos_scores = np.dot(embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(cos_scores)[-top_k:][::-1]
    return df.iloc[top_indices]

def generate_kickoff_packet_from_brief(client_brief, deliverables_df, deliverable_embeddings, inspirations_df, inspiration_embeddings):
    """
    Generates the kick-off packet by sending a clean, well-structured prompt to the AI.
    This version includes final data cleaning for perfect formatting.
    """
    # --- FIND SIMILAR ITEMS ---
    similar_deliverables = find_similar_items(client_brief, deliverables_df, deliverable_embeddings, top_k=3)
    similar_inspirations = find_similar_items(client_brief, inspirations_df, inspiration_embeddings, top_k=5)

    # --- FINAL POLISH (NEW) ---
    # Create copies to avoid SettingWithCopyWarning
    similar_deliverables_clean = similar_deliverables.copy()
    similar_inspirations_clean = similar_inspirations.copy()

    # 1. Replace any NaN values with an empty string for a cleaner table.
    for col in ['Orchestra task name', 'Designer Name', 'Link to Deliverable (Figma ou Frame.io)']:
         if col in similar_deliverables_clean.columns:
            similar_deliverables_clean[col] = similar_deliverables_clean[col].fillna('')

    # 2. Remove newline characters that break the markdown table.
    for col in ['Nom', 'Lien', 'Type']:
        if col in similar_inspirations_clean.columns:
            similar_inspirations_clean[col] = similar_inspirations_clean[col].str.replace('\n', ' ', regex=False).fillna('')


    # --- SELECT COLUMNS AND CREATE PROMPT ---
    deliverable_cols_to_show = ['Orchestra task name', 'Designer Name', 'Link to Deliverable (Figma ou Frame.io)']
    inspiration_cols_to_show = ['Nom', 'Lien', 'Type']
    
    prompt = f"""
    You are "OrchestraAI," an expert AI assistant. Your role is to analyze a new client brief and our internal data to prepare a "Project Kick-off Packet".

    **New Client Brief:**
    "{client_brief}"

    ---
    **Context from our Airtable - Top 3 Similar Past Deliverables:**
    {similar_deliverables_clean[deliverable_cols_to_show].to_markdown(index=False)}
    ---
    **Context from our Airtable - Top 5 Relevant Visual Inspirations:**
    {similar_inspirations_clean[inspiration_cols_to_show].to_markdown(index=False)}
    ---

    **Your Task:**
    Based on the new brief and the provided context tables, generate a "Project Kick-off Packet" in Markdown. The packet must include:
    1.  A brief, one-paragraph analysis of the new project's goals.
    2.  A section titled "## 🚀 Recommended Past Projects" that contains the exact table of deliverables from the context. Below the table, suggest which designer to contact based on the data.
    3.  A section titled "## 🎨 Visual Inspiration" that contains the exact table of inspirations from the context.
    4.  A section titled "## 🤔 Brief Completeness Check" with 3-4 clarifying questions for the client to ensure the designer has all necessary information.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Error during Gemini API call: {e}")
        return f"Error: Could not generate response from Gemini AI. Details: {e}"

# Load data once on server startup
DELIVERABLES_DF, DELIVERABLE_EMBEDDINGS, INSPIRATIONS_DF, INSPIRATION_EMBEDDINGS = get_and_embed_data()
print("✅ Server is ready and listening for webhooks.")

# --- 4. FLASK WEB SERVER ---
app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook_listener():
    """Listens for incoming webhooks."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n🔔 Webhook received at {timestamp}")
    
    data = request.json
    
    if data and data.get('event') == 'task.assigned':
        task_title = data.get('data', {}).get('title', '')
        
        if not task_title:
            print("   -> Event was 'task.assigned', but no title found.")
            return jsonify({"status": "error", "message": "No task title provided"}), 400

        print(f"   -> Processing new task: '{task_title}'")
        
        kickoff_packet = generate_kickoff_packet_from_brief(
            task_title,
            DELIVERABLES_DF,
            DELIVERABLE_EMBEDDINGS,
            INSPIRATIONS_DF,
            INSPIRATION_EMBEDDINGS
        )
        
        print("\n--- 🚀 Generated Kick-off Packet 🚀 ---")
        print(kickoff_packet)
        print("-------------------------------------\n")
        
        return jsonify({"status": "success", "message": "Kick-off packet generated."}), 200
    
    elif data and data.get('event') == 'test.ping':
        print("   -> Received a test ping. Connection is working!")
        return jsonify({"status": "success", "message": "Ping received successfully."}), 200
        
    else:
        print("   -> Received an event, but it was not 'task.assigned' or 'test.ping'. Ignoring.")
        return jsonify({"status": "ignored"}), 200

if __name__ == '__main__':
    # Use a specific port, e.g., 5001 to avoid conflicts if you run other servers
    app.run(host='0.0.0.0', port=5000)