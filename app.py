# app.py
# The final, enhanced Streamlit application for OrchestraAI

import os
import numpy as np
import pandas as pd
from pyairtable import Api
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import streamlit as st
import streamlit.components.v1 as components
import time
import json

# --- 1. CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="OrchestraAI", layout="wide")

st.title("ðŸš€ OrchestraAI Designer's Copilot")
st.markdown("This tool analyzes a new client brief, retrieves relevant context from our Airtable knowledge base, and prepares a comprehensive Project Analysis.")

# Use Streamlit secrets for robust key management
AIRTABLE_API_KEY = "patsNSk4hUHOBMFlM.bf2a4a188bc372286b5196272e394bf4eba94aa0c3e8acb79a8742efb8a34404"
AIRTABLE_BASE_ID = "appU2kDCpzCTLxrNp"
GEMINI_API_KEY = "AIzaSyA4Hpj1KgK0Sm8wH1GfDxpFuMXdmgSVqGE"
DELIVERABLES_TABLE_NAME = "Deliverables"
INSPIRATIONS_TABLE_NAME = "Inspirations"

# --- 2. SETUP THE AI AND DATA CONNECTIONS (CACHED) ---
@st.cache_resource
def get_ai_models():
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return gemini_model, embedding_model

@st.cache_data
def get_airtable_data():
    airtable_api = Api(AIRTABLE_API_KEY)
    
    deliverables_table = airtable_api.table(AIRTABLE_BASE_ID, DELIVERABLES_TABLE_NAME)
    all_deliverables = deliverables_table.all()
    deliverables_df = pd.DataFrame([record['fields'] for record in all_deliverables])
    deliverables_df.dropna(how='all', inplace=True)
    deliverables_df.fillna('', inplace=True)
    deliverables_df['search_text'] = deliverables_df['Orchestra task name'] + ". " + deliverables_df['Notes on Timing (if Late/Early)']

    inspirations_table = airtable_api.table(AIRTABLE_BASE_ID, INSPIRATIONS_TABLE_NAME)
    all_inspirations = inspirations_table.all()
    inspirations_df = pd.DataFrame([record['fields'] for record in all_inspirations])
    inspirations_df.dropna(how='all', inplace=True)
    inspirations_df.fillna('', inplace=True)
    inspirations_df['search_text'] = inspirations_df['Nom'] + ". Tags: " + inspirations_df['Type']
    
    return deliverables_df, inspirations_df

gemini_model, embedding_model = get_ai_models()
deliverables_df, inspirations_df = get_airtable_data()

# --- 3. CORE LOGIC (EXISTING & NEW) ---
@st.cache_data
def get_embeddings(_df, column_name):
    return embedding_model.encode(_df[column_name].tolist())

deliverable_embeddings = get_embeddings(deliverables_df, 'search_text')
inspiration_embeddings = get_embeddings(inspirations_df, 'search_text')

def find_similar_items(query, df, embeddings, top_k=5):
    query_embedding = embedding_model.encode([query])
    cos_scores = np.dot(embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(cos_scores)[-top_k:][::-1]
    return df.iloc[top_indices]

# === NEW ENHANCED FUNCTIONS (FROM SENIOR) ===

def extract_project_attributes(brief, gemini_model):
    """Extract structured attributes from the brief using AI"""
    prompt = f"""
    Analyze this client brief and extract key attributes in JSON format:
    "{brief}"
    
    Return ONLY a valid JSON object with these keys:
    {{
        "industry": "detected industry",
        "project_type": "website/app/branding/etc",
        "complexity": "simple/medium/complex",
        "timeline": "any mentioned deadlines",
        "budget_signals": "any cost indicators",
        "key_features": ["list", "of", "features"],
        "tone": "modern/classic/playful/professional"
    }}
    """
    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:-3]
        elif response_text.startswith('```'):
            response_text = response_text[3:-3]
        return json.loads(response_text)
    except:
        return {"error": "Failed to parse JSON attributes."}

def analyze_designer_expertise(deliverables_df):
    """Build designer expertise profiles"""
    designer_profiles = {}
    for designer in deliverables_df['Designer Name'].unique():
        if pd.isna(designer) or designer == '':
            continue
        designer_work = deliverables_df[deliverables_df['Designer Name'] == designer]
        task_names = designer_work['Orchestra task name'].fillna('')
        specializations = []
        all_tasks = ' '.join(task_names.str.lower())
        if 'e-commerce' in all_tasks or 'shop' in all_tasks: specializations.append('E-commerce')
        if 'dashboard' in all_tasks: specializations.append('Dashboards')
        if 'mobile' in all_tasks or 'app' in all_tasks: specializations.append('Mobile')
        if 'branding' in all_tasks or 'logo' in all_tasks: specializations.append('Branding')
        if 'website' in all_tasks or 'web' in all_tasks: specializations.append('Web Design')
        
        designer_profiles[designer] = {
            'total_projects': len(designer_work),
            'specializations': list(set(specializations)),
            'all_work_df': designer_work # Store all their work for later
        }
    return designer_profiles

def recommend_designers(brief, designer_profiles, top_k=3):
    """Recommend best designers for the project"""
    brief_lower = brief.lower()
    designer_scores = {}
    for designer, profile in designer_profiles.items():
        score = 0
        score += min(profile['total_projects'] * 0.1, 2.0)
        for spec in profile['specializations']:
            if any(keyword in brief_lower for keyword in spec.lower().split()):
                score += 2.0
        designer_scores[designer] = score
    sorted_designers = sorted(designer_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_designers[:top_k]

def estimate_project_complexity(brief):
    """AI-powered project complexity scoring"""
    complexity_indicators = {'e-commerce': 3, 'payment': 3, 'database': 2, 'api': 3, 'integration': 2, 'mobile app': 4, 'dashboard': 2, 'simple': -1, 'basic': -1, 'landing page': -1}
    brief_lower = brief.lower()
    score = sum(weight for indicator, weight in complexity_indicators.items() if indicator in brief_lower)
    if score <= 1: return "Simple", score
    elif score <= 4: return "Medium", score
    else: return "Complex", score

def generate_project_scope(brief, similar_deliverables, gemini_model):
    """Generate detailed project scope with deliverables breakdown"""
    complexity, score = estimate_project_complexity(brief)
    similar_tasks = similar_deliverables['Orchestra task name'].head(3).tolist()
    prompt = f"""
    As a senior project manager, create a comprehensive project scope for this brief:
    **Client Brief:** "{brief}"
    **Detected Complexity:** {complexity}
    **Similar Past Projects:** {similar_tasks}
    
    Create a detailed scope with these sections:
    ## ðŸŽ¯ Project Overview
    ## ðŸ“‹ Deliverables Breakdown (by phase)
    ## â±ï¸ Timeline Estimate
    ## ðŸ‘¥ Team Requirements
    ## ðŸŽ¯ Success Metrics
    ## âš ï¸ Risk Assessment
    Be specific and actionable.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

def generate_intelligent_questions(brief, similar_deliverables, gemini_model):
    """Generate contextual questions based on what's missing"""
    similar_tasks = similar_deliverables['Orchestra task name'].head(5).tolist()
    prompt = f"""
    You're a senior project manager reviewing this client brief: "{brief}"
    Based on these similar past projects we've done: {similar_tasks}
    Generate 5-7 intelligent clarifying questions. Format as:
    **1. [Category] - [Question]**
    *Why this matters: [Brief explanation]*
    """
    response = gemini_model.generate_content(prompt)
    return response.text

def generate_figma_draft_html(project_scope):
    """ (V2 FEATURE) Takes the generated scope and creates an HTML draft. """
    prompt = f"""
    You are an expert front-end developer. Based on the following "Project Scope", generate a single HTML file for a webpage draft.
    **CRITICAL INSTRUCTIONS:**
    - Use Tailwind CSS for all styling from a CDN.
    - Use semantic HTML5 tags.
    - Use placeholder images from `https://placehold.co/`.
    - The design should be clean and directly reflect the goals in the scope.
    - Output ONLY the HTML code, starting with `<!DOCTYPE html>`.

    ---
    **Project Scope:**
    {project_scope}
    ---
    """
    response = gemini_model.generate_content(prompt)
    html_code = response.text.strip().replace("```html", "").replace("```", "")
    return html_code

# --- 4. STREAMLIT UI ---
client_brief = st.text_area("Enter Client Brief:", height=150, placeholder="e.g., We need a new e-commerce website for our coffee brand...")

if st.button("ðŸš€ Generate Complete Analysis", type="primary"):
    if client_brief:
        progress_bar = st.progress(0, text="Initializing Analysis...")
        
        # Step 1: Basic Analysis
        progress_bar.progress(10, text="ðŸ§  Analyzing brief and extracting attributes...")
        project_attributes = extract_project_attributes(client_brief, gemini_model)
        complexity, complexity_score = estimate_project_complexity(client_brief)
        
        # Step 2: Find Similar Items
        progress_bar.progress(25, text="ðŸ” Finding similar deliverables and inspirations...")
        similar_deliverables = find_similar_items(client_brief, deliverables_df, deliverable_embeddings)
        similar_inspirations = find_similar_items(client_brief, inspirations_df, inspiration_embeddings)
        
        # Step 3: Designer Analysis
        progress_bar.progress(40, text="ðŸ‘¨â€ðŸŽ¨ Analyzing designer expertise...")
        designer_profiles = analyze_designer_expertise(deliverables_df)
        recommended_designers = recommend_designers(client_brief, designer_profiles)
        
        # Step 4: Generate Content
        progress_bar.progress(60, text="ðŸ“ Generating project scope and intelligent questions...")
        project_scope = generate_project_scope(client_brief, similar_deliverables, gemini_model)
        intelligent_questions = generate_intelligent_questions(client_brief, similar_deliverables, gemini_model)
        
        # Step 5: Generate HTML Draft
        progress_bar.progress(85, text="ðŸŽ¨ Creating Figma-ready HTML draft...")
        figma_html = generate_figma_draft_html(project_scope)
        
        progress_bar.progress(100, text="âœ… Analysis Complete!")
        time.sleep(1)
        progress_bar.empty()

        # --- RESULTS DISPLAY ---
        st.divider()
        st.header("ðŸŽ¯ Complete Project Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Project Complexity", complexity, f"Score: {complexity_score:.1f}")
        col2.metric("Similar Projects Found", len(similar_deliverables))
        col3.metric("Recommended Designers", len(recommended_designers))
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["ðŸ“‹ Project Scope", "â“ Intelligent Questions", "ðŸ‘¥ Designer Recommendations", "ðŸŽ¨ Visual Inspiration", "ðŸ–¼ï¸ Figma Draft"])
        
        with tab1:
            st.markdown(project_scope)
            with st.expander("ðŸ” Detected Project Attributes (JSON)"):
                st.json(project_attributes)
        
        with tab2:
            st.markdown(intelligent_questions)
            with st.expander("ðŸ“Š Based on Similar Past Projects"):
                st.dataframe(similar_deliverables[['Orchestra task name', 'Designer Name', 'Notes on Timing (if Late/Early)']])
        
        with tab3:
            for i, (designer, score) in enumerate(recommended_designers, 1):
                profile = designer_profiles.get(designer, {})
                st.markdown(f"### {i}. {designer}")
                
                # Display metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Match Score", f"{score:.1f}")
                c2.metric("Total Projects", profile.get('total_projects', 0))
                c3.metric("Specializations", len(profile.get('specializations', [])))
                if profile.get('specializations'):
                    st.write("**Specialties:** " + ", ".join(profile['specializations']))

                # --- NEW FEATURE: Find and display most relevant projects for THIS designer ---
                with st.expander(f"Show most relevant projects by {designer}"):
                    designer_work_df = profile.get('all_work_df')
                    if designer_work_df is not None and not designer_work_df.empty:
                        # We need to get embeddings for just this designer's work
                        designer_embeddings = get_embeddings(designer_work_df, 'search_text')
                        relevant_projects = find_similar_items(client_brief, designer_work_df, designer_embeddings, top_k=2)
                        
                        st.write(f"**Top {len(relevant_projects)} most relevant projects for this brief:**")
                        
                        for _, project_row in relevant_projects.iterrows():
                            st.markdown(f"**Project:** {project_row['Orchestra task name']}")
                            st.markdown(f"  - **Figma:** {project_row['Link to Deliverable (Figma ou Frame.io)']}")
                            st.markdown(f"  - **Orchestra:** {project_row['ðŸŒ¸ Orchestra task link']}")
                            st.markdown(f"  - **Claap Video:** {project_row['Link to Claap Video']}")
                    else:
                        st.write("No projects found for this designer.")
                st.divider()
        
        with tab4:
            st.dataframe(similar_inspirations[['Nom', 'Type', 'Lien']])
        
        with tab5:
            subtab1, subtab2 = st.tabs(["ðŸ“„ HTML Code", "ðŸ–¼ï¸ Live Preview"])
            with subtab1:
                st.code(figma_html, language="html")
            with subtab2:
                components.html(figma_html, height=600, scrolling=True)
    else:
        st.warning("Please enter a client brief to begin analysis.")
