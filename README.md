🚀 OrchestraAI Designer's Copilot

OrchestraAI is a proof-of-concept AI agent designed to accelerate a designer's workflow. It analyzes a new client brief, retrieves relevant historical context and visual inspiration from a knowledge base, and generates a comprehensive "Project Kick-off Packet." This tool demonstrates a complete V2 vision, including the automated generation of a preliminary Figma draft using HTML.
✨ Features

    AI-Powered Brief Analysis: Uses Google's Gemini model to understand the core objectives of a new client brief.

    Retrieval-Augmented Generation (RAG): Connects to an Airtable database to find and retrieve relevant past projects and visual inspirations.

    Semantic Search: Employs sentence-transformer models to find contextually similar data, going beyond simple keyword matching.

    Brief Completeness Check: Automatically identifies missing information in a brief and generates clarifying questions for the client.

    V2 Figma Draft Generation: Creates a preliminary webpage design in HTML with Tailwind CSS, which can be directly imported into Figma using plugins like "HTML to Design."

    Interactive Web UI: Built with Streamlit for a user-friendly and professional interface.

🛠️ Tech Stack

    Backend & Logic: Python

    Web Framework: Streamlit

    AI Model: Google Gemini 1.5 Flash

    Semantic Search: sentence-transformers (all-MiniLM-L6-v2)

    Database: Airtable (via py-airtable)

    Data Handling: Pandas

⚙️ Setup and Installation

Follow these steps to get the OrchestraAI Copilot running on your local machine.
1. Prerequisites

    Python 3.9+

    An Airtable account with a Base set up according to the project's data structure.

2. Clone the Repository

git clone [https://github.com/your-username/OrchestraAI.git](https://github.com/your-username/OrchestraAI.git)
cd OrchestraAI

3. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

# For Windows
python -m venv .venv
.\.venv\Scripts\activate

# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

4. Install Dependencies

Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt

5. Configure API Keys

The application uses API keys for Airtable and Google Gemini. These are currently hard-coded in the app.py script. For a production environment, it is recommended to manage these using Streamlit's secrets management.
▶️ How to Run

Once the setup is complete, you can run the Streamlit application with a single command:

streamlit run app.py

Your web browser will automatically open a new tab with the application running.
🔬 How It Works

    Data Ingestion: On startup, the app connects to the specified Airtable base and fetches all records from the Deliverables and Inspirations tables. It then cleans this data.

    Embedding Generation: The text from these records is converted into numerical vectors (embeddings) using a sentence-transformer model. This process happens only once and the results are cached for speed.

    User Input: The user pastes a new client brief into the text area.

    Similarity Search: When the user clicks "Generate," the application creates an embedding for the client brief and compares it against the cached embeddings to find the most similar past projects and inspirations.

    Packet Generation (AI Call 1): The client brief and the retrieved data are formatted into a detailed prompt and sent to the Gemini API, which generates the markdown-based "Project Kick-off Packet."

    Figma Draft (AI Call 2): The generated packet is then used as context for a second prompt, which instructs the Gemini API to create an HTML/Tailwind CSS prototype of a webpage.

    Display: The final Kick-off Packet and the interactive HTML draft are displayed in the Streamlit interface.