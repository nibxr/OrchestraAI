# app.py
# Enhanced OrchestraAI Copilot with a Professional and Cohesive UI

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

# --- CONFIGURATION & BEAUTIFUL PAGE SETUP ---
st.set_page_config(
    page_title="🎼 OrchestraAI Copilot",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon="🎼"
)

# Custom CSS for the new redesigned UI
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Root Variables */
    :root {
        --primary-color: #5A5AFF;
        --primary-hover: #4A4AFF;
        --secondary-color: #F5F5F7;
        --accent-color: #00E5A1;
        --text-dark: #1D1D1F;
        --text-light: #6E6E73;
        --border-color: #D2D2D7;
        --gradient: linear-gradient(135deg, #5A5AFF 0%, #A25AFF 100%);
    }

    /* Main App Container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }

    /* Custom Header */
    .custom-header {
        background: var(--gradient);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(90, 90, 255, 0.2);
        text-align: center;
        color: white;
    }

    .custom-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
    }

    .custom-header p {
        font-size: 1.3rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }

    /* Beautiful Cards */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
    }

    .card-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-dark);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .card-description {
        color: var(--text-light);
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    /* Modern Buttons */
    .stButton > button {
        background: var(--primary-color) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background: var(--primary-hover) !important;
        transform: translateY(-2px) !important;
    }

    /* Text Areas and Inputs */
    .stTextArea textarea, .stTextInput input {
        border-radius: 12px !important;
        border: 1px solid var(--border-color) !important;
        padding: 1rem !important;
        font-family: 'Inter', sans-serif !important;
        background-color: #FFFFFF !important;
        color: var(--text-dark) !important;
    }

    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(90, 90, 255, 0.15) !important;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background: var(--secondary-color);
        border-right: 1px solid var(--border-color);
    }

    .sidebar-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--text-dark);
        margin-bottom: 1rem;
        padding: 1rem;
        background: white;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Metrics and Stats */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin: 0.5rem 0;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary-color);
        font-family: 'Inter', sans-serif;
    }

    .metric-label {
        color: var(--text-light);
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--secondary-color);
        padding: 0.5rem;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border: none;
        background: transparent;
        color: var(--text-light);
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: white !important;
        color: var(--text-dark) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Visible Sidebar Toggle Button */
    button[data-testid="stSidebarNavCollapseButton"] {
        display: inline-block;
        padding: 0.5rem;
        margin-left: 0.5rem;
        background-color: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }

    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# --- API CONFIGURATION ---
try:
    AIRTABLE_API_KEY = st.secrets["AIRTABLE_API_KEY"]
    AIRTABLE_BASE_ID = st.secrets["AIRTABLE_BASE_ID"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    print("API key issuem, check the file")

DELIVERABLES_TABLE_NAME = "Deliverables"
INSPIRATIONS_TABLE_NAME = "Inspirations"

# --- CACHED FUNCTIONS ---
@st.cache_resource
def get_ai_models():
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return gemini_model, embedding_model

@st.cache_data
def get_airtable_data():
    try:
        airtable_api = Api(AIRTABLE_API_KEY)

        # Fetch Deliverables
        deliverables_table = airtable_api.table(AIRTABLE_BASE_ID, DELIVERABLES_TABLE_NAME)
        all_deliverables = deliverables_table.all()
        deliverables_df = pd.DataFrame([record['fields'] for record in all_deliverables])
        deliverables_df.dropna(how='all', inplace=True)
        deliverables_df.fillna('', inplace=True)
        if 'Orchestra task name' in deliverables_df.columns and 'Notes on Timing (if Late/Early)' in deliverables_df.columns:
            deliverables_df['search_text'] = deliverables_df['Orchestra task name'] + ". " + deliverables_df['Notes on Timing (if Late/Early)']

        # Fetch Inspirations
        inspirations_table = airtable_api.table(AIRTABLE_BASE_ID, INSPIRATIONS_TABLE_NAME)
        all_inspirations = inspirations_table.all()
        inspirations_df = pd.DataFrame([record['fields'] for record in all_inspirations])
        inspirations_df.dropna(how='all', inplace=True)
        inspirations_df.fillna('', inplace=True)
        if 'Nom' in inspirations_df.columns and 'Type' in inspirations_df.columns:
            inspirations_df['search_text'] = inspirations_df['Nom'] + ". Tags: " + inspirations_df['Type']

        return deliverables_df, inspirations_df
    except Exception as e:
        st.error(f"Failed to connect to Airtable: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Load models and data
gemini_model, embedding_model = get_ai_models()
deliverables_df, inspirations_df = get_airtable_data()

# --- CORE FUNCTIONS ---
@st.cache_data
def get_embeddings(_df, column_name='search_text'):
    if column_name in _df.columns and not _df.empty:
        return embedding_model.encode(_df[column_name].tolist())
    return np.array([])

if not deliverables_df.empty:
    deliverable_embeddings = get_embeddings(deliverables_df)
if not inspirations_df.empty:
    inspiration_embeddings = get_embeddings(inspirations_df)

def find_similar_items(query, df, embeddings, top_k=5):
    if embeddings.size == 0 or df.empty:
        return pd.DataFrame()
    query_embedding = embedding_model.encode([query])
    cos_scores = np.dot(embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(cos_scores)[-top_k:][::-1]
    return df.iloc[top_indices]

def extract_project_attributes(brief, gemini_model):
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
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```"):
            response_text = response_text[3:-3].strip()
        return json.loads(response_text)
    except Exception:
        return {"error": "Failed to parse JSON attributes from the AI response."}

def estimate_project_complexity(brief):
    complexity_indicators = {
        'e-commerce': 3, 'payment': 3, 'database': 2, 'api': 3,
        'integration': 2, 'mobile app': 4, 'dashboard': 2, 'animation': 2,
        'simple': -1, 'basic': -1, 'landing page': -1, 'brochure': -1
    }
    brief_lower = brief.lower()
    score = sum(weight for indicator, weight in complexity_indicators.items() if indicator in brief_lower)
    if score <= 1: return "Simple", score
    if score <= 4: return "Medium", score
    return "Complex", score

def analyze_designer_expertise(deliverables_df):
    designer_profiles = {}
    if 'Designer Name' not in deliverables_df.columns:
        return {}
    for designer in deliverables_df['Designer Name'].unique():
        if pd.isna(designer) or designer == '': continue
        designer_work = deliverables_df[deliverables_df['Designer Name'] == designer]
        task_names = designer_work['Orchestra task name'].fillna('')
        specializations = []
        all_tasks = ' '.join(task_names.str.lower())
        if 'e-commerce' in all_tasks or 'shop' in all_tasks: specializations.append('E-commerce')
        if 'dashboard' in all_tasks: specializations.append('Dashboards')
        if 'mobile' in all_tasks or 'app' in all_tasks: specializations.append('Mobile Apps')
        if 'branding' in all_tasks or 'logo' in all_tasks: specializations.append('Branding')
        if 'website' in all_tasks or 'web' in all_tasks: specializations.append('Web Design')
        designer_profiles[designer] = {
            'total_projects': len(designer_work),
            'specializations': list(set(specializations)) if specializations else ['Generalist'],
            'all_work_df': designer_work
        }
    return designer_profiles

def recommend_designers(brief, designer_profiles, top_k=3):
    brief_lower = brief.lower()
    designer_scores = {}
    for designer, profile in designer_profiles.items():
        score = np.log1p(profile['total_projects'])
        for spec in profile['specializations']:
            if any(keyword in brief_lower for keyword in spec.lower().split()):
                score += 2.5
        designer_scores[designer] = score
    return sorted(designer_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

def generate_project_scope(brief, similar_deliverables, gemini_model):
    complexity, _ = estimate_project_complexity(brief)
    similar_tasks_list = similar_deliverables['Orchestra task name'].head(3).tolist() if not similar_deliverables.empty else "None"
    prompt = f"""
    As a senior project manager, create a comprehensive project scope for this brief.
    **Client Brief:** "{brief}"
    **Detected Complexity:** {complexity}
    **Similar Past Projects:** {similar_tasks_list}
    Create a detailed scope with these markdown sections:
    ### 🎯 Project Overview
    ### 📋 Deliverables Breakdown
    ### ⏳ Timeline Estimate
    ### ✅ Success Metrics
    ### ⚠️ Potential Risks & Mitigations
    """
    response = gemini_model.generate_content(prompt)
    return response.text

def generate_intelligent_questions(brief, similar_deliverables, gemini_model):
    similar_tasks_list = similar_deliverables['Orchestra task name'].head(5).tolist() if not similar_deliverables.empty else "None"
    prompt = f"""
    You are a senior design strategist reviewing this client brief: "{brief}"
    Similar past projects: {similar_tasks_list}
    Generate 5-6 insightful questions to ask the client. Format each as:
    **1. Category - [Your Question]**
    * **Why it matters:** [Brief explanation]*
    """
    response = gemini_model.generate_content(prompt)
    return response.text

def generate_figma_draft_html(project_scope):
    prompt = f"""
    You are a world-class UI/UX designer and front-end developer tasked with creating a visually stunning and comprehensive multi-section landing page draft based on a project scope.

    **Project Scope:**
    ---
    {project_scope}
    ---

    **CRITICAL INSTRUCTIONS:**

    1.  **Generate a COMPLETE, MULTI-SECTION WEBPAGE:** The output must be a full landing page, not just one section. Include the following sections, adapted to the project scope:
        * **Header/Navigation Bar:** With a placeholder logo and navigation links.
        * **Hero Section:** A large, impactful introduction with a clear headline, subheadline, and a primary call-to-action button.
        * **Features/Services Section:** A detailed breakdown of the key features or services mentioned in the scope. Use a grid layout with icons and short descriptions.
        * **About Us/Our Story Section:** A section that tells the brand's story, reflecting the tone from the project scope.
        * **Testimonials Section:** Include 2-3 placeholder testimonials with names and titles.
        * **Call to Action (CTA) Section:** A final, compelling CTA to encourage user action.
        * **Footer:** With links, social media icons, and a copyright notice.

    2.  **STYLING (TAILWIND CSS):**
        * Use Tailwind CSS for ALL styling, loaded from the official CDN (`<script src="https://cdn.tailwindcss.com"></script>`).
        * Apply modern design principles. Use a professional and consistent color scheme (e.g., a primary color, grays for text, and an accent color).
        * Ensure ample whitespace for a clean, uncluttered look.
        * The design must be fully responsive and look great on desktop, tablet, and mobile devices.

    3.  **CONTENT:**
        * Write compelling, relevant placeholder copy for all sections. Do NOT use "Lorem Ipsum." The content should directly relate to the project described in the scope.
        * Use placeholder images from `https://placehold.co/` (e.g., `https://placehold.co/600x400`).
        * Make statistic Graphics or Graphs whevever possible.

    4.  **OUTPUT:**
        * Respond with ONLY the complete HTML code.
        * The code must start with `<!DOCTYPE html>` and end with `</html>`.
        * Do not include any explanations, markdown formatting like ```html, or any text outside of the HTML code itself.
    """
    response = gemini_model.generate_content(prompt)
    html_code = response.text.strip()
    if html_code.startswith("```html"):
        html_code = html_code[7:-3]
    elif html_code.startswith("```"):
        html_code = html_code[3:-3]
    return html_code

# --- BEAUTIFUL UI COMPONENTS ---
def create_header():
    st.markdown("""
    <div class="custom-header">
        <h1>🎼 OrchestraAI Copilot</h1>
        <p>Intelligent Project Analysis & Planning</p>
    </div>
    """, unsafe_allow_html=True)

def create_metrics_dashboard():
    if not deliverables_df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(deliverables_df)}</div><div class="metric-label">Total Projects</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(inspirations_df)}</div><div class="metric-label">Inspirations</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">AI</div><div class="metric-label">Powered</div></div>', unsafe_allow_html=True)

def create_feature_card(title, description, icon="🚀"):
    return f"""
    <div class="feature-card">
        <div class="card-title">{icon} {title}</div>
        <div class="card-description">{description}</div>
    </div>
    """

# --- MAIN APPLICATION ---
def main():
    create_header()

    # Initialize session state for history
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

    with st.sidebar:
        st.markdown('<div class="sidebar-title">🎯 Navigation</div>', unsafe_allow_html=True)
        page = st.selectbox("Choose a page:", ["🏠 Dashboard", "📝 Project Analysis", "📜 History", "🔍 Search Projects", "💡 Inspirations"])

    if page == "🏠 Dashboard":
        st.markdown("## Welcome to Your Project Command Center")
        create_metrics_dashboard()
        st.markdown("### 🚀 Quick Actions")
        st.markdown(create_feature_card("Smart Project Analysis", "Get AI-powered insights, complexity assessment, and scope recommendations for any client brief.", "🧠"), unsafe_allow_html=True)
        
        with st.expander("💡 Need inspiration? Try these example prompts!"):
            st.markdown("""
            - "We're a new sustainable fashion brand targeting millennials. We need a full branding package, including a logo, color palette, and an e-commerce website."
            - "Design a mobile app dashboard for a smart home system. It should be clean, minimalist, and allow users to control lighting, temperature, and security."
            - "Create a modern, professional website for a B2B SaaS company that sells project management software. The site needs to highlight key features, have a pricing page, and a blog."
            """)

    elif page == "📝 Project Analysis":
        st.markdown("## 🧠 Intelligent Project Analysis")
        st.markdown("Transform client briefs into actionable project plans with AI insights.")
        st.markdown(create_feature_card("Client Brief Input", "Paste your client brief below and our AI will analyze it.", "📝"), unsafe_allow_html=True)
        client_brief = st.text_area("Enter Client Brief:", placeholder="Paste the client brief here...", height=150, key="brief_input")

        if st.button("🚀 Analyze Project", type="primary") and client_brief:
            if deliverables_df.empty or inspirations_df.empty:
                st.error("Data could not be loaded from Airtable. Please check configurations.")
            else:
                progress_bar = st.progress(0, text="Initializing Analysis...")
                
                # Step 1: Basic Analysis
                progress_bar.progress(10, text="🧠 Analyzing brief and extracting attributes...")
                project_attributes = extract_project_attributes(client_brief, gemini_model)
                complexity, complexity_score = estimate_project_complexity(client_brief)
                
                # Step 2: Similarity Search
                progress_bar.progress(25, text="🔎 Finding similar projects and inspirations...")
                similar_deliverables = find_similar_items(client_brief, deliverables_df, deliverable_embeddings)
                similar_inspirations = find_similar_items(client_brief, inspirations_df, inspiration_embeddings)
                
                # Step 3: Designer Analysis
                progress_bar.progress(40, text="🧑‍🎨 Analyzing designer expertise...")
                designer_profiles = analyze_designer_expertise(deliverables_df)
                recommended_designers = recommend_designers(client_brief, designer_profiles)
                
                # Step 4: Content Generation
                progress_bar.progress(60, text="✍️ Generating detailed project scope...")
                project_scope = generate_project_scope(client_brief, similar_deliverables, gemini_model)
                
                progress_bar.progress(75, text="❓ Generating clarifying questions...")
                intelligent_questions = generate_intelligent_questions(client_brief, similar_deliverables, gemini_model)
                
                # Step 5: HTML Draft Generation
                progress_bar.progress(90, text="🎨 Creating a Figma-ready HTML draft...")
                figma_html = generate_figma_draft_html(project_scope)
                
                progress_bar.progress(100, text="✅ Analysis Complete!")
                time.sleep(1)
                progress_bar.empty()

                # Store the analysis in history
                history_entry = {
                    "brief": client_brief,
                    "scope": project_scope,
                    "questions": intelligent_questions,
                    "designers": recommended_designers,
                    "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
                }
                st.session_state.analysis_history.insert(0, history_entry)

                # --- NEW RESULTS DISPLAY ---
                st.markdown("---")
                st.subheader("High-Level Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Project Complexity", complexity, f"Score: {complexity_score:.1f}")
                with col2:
                    st.metric("Similar Projects Found", len(similar_deliverables) if not similar_deliverables.empty else 0)
                with col3:
                    st.metric("Top Designers Recommended", len(recommended_designers))

                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Project Scope", "❓ Clarifying Questions", "🧑‍🎨 Designer Recommendations", "🎨 Visual Inspiration", "🌐 HTML Draft"])
                
                with tab1:
                    st.header("📋 Detailed Project Scope")
                    st.markdown(project_scope)
                    with st.expander("🔍 Detected Project Attributes (JSON View)"):
                        st.json(project_attributes)
                
                with tab2:
                    st.header("❓ Intelligent Clarifying Questions")
                    st.markdown("Use these questions to guide your client kick-off call.")
                    st.markdown(intelligent_questions)
                    with st.expander("📚 Based on These Similar Past Projects"):
                        st.dataframe(similar_deliverables[['Orchestra task name', 'Designer Name', 'Notes on Timing (if Late/Early)']])
                
                with tab3:
                    st.header("🧑‍🎨 Top Designer Recommendations")
                    st.markdown("These designers are recommended based on their past work's relevance.")
                    for i, (designer, score) in enumerate(recommended_designers, 1):
                        profile = designer_profiles.get(designer, {})
                        with st.container(border=True):
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.subheader(f"{i}. {designer}")
                                st.write("**Specialties:** " + ", ".join(profile.get('specializations', ['N/A'])))
                            with c2:
                                st.metric("Match Score", f"{score:.2f}")
                            with st.expander(f"Show most relevant projects by {designer}"):
                                st.dataframe(profile.get('all_work_df', pd.DataFrame()))
                
                with tab4:
                    st.header("🎨 Visual Inspiration & Similar Projects")
                    st.markdown("Here are some existing designs that share similarities with the brief.")
                    for _, row in similar_inspirations.iterrows():
                        with st.container(border=True):
                            st.subheader(row.get('Nom', 'N/A'))
                            st.write(f"**Type:** {row.get('Type', 'N/A')}")
                            if row.get('Lien'):
                                st.link_button("🔗 View Inspiration", row['Lien'])
                
                with tab5:
                    st.header("🌐 Figma-Ready HTML Draft")
                    st.markdown("A basic HTML structure based on the scope for rapid prototyping.")
                    subtab1, sub2 = st.tabs(["🖥️ Live Preview", " </> HTML Code"])
                    with subtab1:
                        components.html(figma_html, height=600, scrolling=True)
                    with sub2:
                        st.code(figma_html, language="html")

    elif page == "📜 History":
        st.markdown("## 📜 Analysis History")
        st.markdown("Review your previously analyzed client briefs and their results.")

        if not st.session_state.analysis_history:
            st.info("You haven't analyzed any briefs yet. Go to the 'Project Analysis' page to get started.")
        else:
            for i, entry in enumerate(st.session_state.analysis_history):
                with st.expander(f"**{entry['timestamp']}** - Brief: {entry['brief'][:50]}..."):
                    st.markdown(f"#### Client Brief")
                    st.info(entry['brief'])
                    st.markdown("---")
                    st.markdown("#### 📋 Detailed Project Scope")
                    st.markdown(entry['scope'])
                    st.markdown("---")
                    st.markdown("#### ❓ Clarifying Questions")
                    st.markdown(entry['questions'])
                    st.markdown("---")
                    st.markdown("#### 🧑‍🎨 Top Designer Recommendations")
                    for designer, score in entry['designers']:
                        st.markdown(f"- **{designer}** (Match Score: {score:.2f})")


    elif page == "🔍 Search Projects":
        st.markdown("## 🔍 Search Past Projects")
        st.markdown("Search through your project database using natural language.")
        search_query = st.text_input("Search Projects:", placeholder="e.g., mobile app design, e-commerce website")
        if search_query and not deliverables_df.empty:
            similar_projects = find_similar_items(search_query, deliverables_df, deliverable_embeddings, top_k=10)
            st.markdown(f"### Found {len(similar_projects)} Similar Projects")
            st.dataframe(similar_projects)
        st.markdown("---")
        st.markdown("### All Past Projects from Airtable")
        st.dataframe(deliverables_df)

    elif page == "💡 Inspirations":
        st.markdown("## 💡 Design Inspirations")
        st.markdown("Browse and search through design inspirations.")
        if not inspirations_df.empty:
            search_inspiration = st.text_input("Search Inspirations:", placeholder="e.g., modern, minimalist, colorful")
            if search_inspiration:
                st.markdown("### Search Results")
                similar_inspirations = find_similar_items(search_inspiration, inspirations_df, inspiration_embeddings, top_k=10)
                if not similar_inspirations.empty:
                    cols = st.columns(2)
                    for idx, row in similar_inspirations.iterrows():
                        col = cols[idx % 2]
                        with col:
                            with st.container(border=True):
                                st.subheader(row.get('Nom', 'N/A'))
                                st.write(f"**Type:** {row.get('Type', 'N/A')}")
                                if row.get('Lien'):
                                    st.link_button("🔗 View Inspiration", row['Lien'])
                else:
                    st.info("No inspirations found matching your search.")
            st.markdown("---")
            st.markdown("### All Inspirations from Airtable")
            st.dataframe(inspirations_df)
        else:
            st.warning("No inspirations data available from Airtable.")

if __name__ == "__main__":
    main()
