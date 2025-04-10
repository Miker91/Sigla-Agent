# BUSINESS SCIENCE
# Pandas Data Analyst App
# -----------------------

# This app is designed to help you analyze data and create data visualizations from natural language requests.

# Imports
# !pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from openai import OpenAI
import os
import sys
import streamlit as st
import pandas as pd
import plotly.io as pio
import json
import yaml
import io
import tempfile
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

# LangSmith imports for observability
from langsmith import Client, traceable
from langsmith.wrappers import wrap_openai
import uuid

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from ai_data_science_team import (
    PandasDataAnalyst,
    DataWranglingAgent,
    DataVisualizationAgent,
    FeatureEngineeringAgent,
    DataCleaningAgent,
)

# Import custom agents
from agents import (
    ConfigurationAnalysisAgent,
    DataChatAgent,
)

# Try to find .env file in multiple possible locations
possible_paths = [
    os.path.join(os.path.dirname(__file__), '..', '..', '.env'),  # Two directories up
    os.path.join(os.path.dirname(__file__), '..', '.env'),        # One directory up
    os.path.join(os.path.dirname(__file__), '.env'),              # Same directory
    '.env'                                                         # Current working directory
]

dotenv_path = next((path for path in possible_paths if os.path.exists(path)), 
                   os.path.join(os.path.dirname(__file__), '..', '..', '.env'))  # Default if none found
load_dotenv(dotenv_path)

# Setup LangSmith for observability
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "pandas-data-analyst"

# Check if LANGSMITH_API_KEY is set, otherwise try to get from environment
if not os.getenv("LANGSMITH_API_KEY"):
    langsmith_api_key = os.getenv("OPENAI_API_KEY")  # Fallback to using OpenAI key
    if langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = langsmith_api_key

# Initialize LangSmith client
try:
    ls_client = Client()
    LANGSMITH_ENABLED = True
except Exception as e:
    st.sidebar.warning(f"LangSmith initialization failed: {e}. Observability will be limited.")
    LANGSMITH_ENABLED = False

# * APP INPUTS ----

MODEL_LIST = ["gpt-4o-mini", "gpt-4o"]
TITLE = "Analityk danych Sigla"

# Application modes
APP_MODES = ["Analiza danych z czatem", "Analiza z pliku konfiguracyjnego", "Czatowanie z danymi"]

# Sample data generator
def create_sample_bike_sales_data():
    """Generate sample bike sales data for demonstration purposes."""
    np.random.seed(42)
    
    # Define parameters
    n_rows = 500
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2023-12-31')
    
    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    selected_dates = np.random.choice(dates, size=n_rows, replace=True)
    selected_dates = sorted(selected_dates)
    
    # Bike models
    models = ['Mountain Pro', 'City Cruiser', 'Road Elite', 'Explorer', 'Kids Fun']
    
    # Bike colors
    colors = ['Red', 'Blue', 'Green', 'Black', 'White', 'Yellow', 'Silver']
    
    # Categories
    categories = ['Budget', 'Standard', 'Premium']
    
    # Generate data
    data = {
        'date': selected_dates,
        'model': np.random.choice(models, size=n_rows),
        'color': np.random.choice(colors, size=n_rows),
        'category': np.random.choice(categories, size=n_rows, p=[0.2, 0.5, 0.3]),
    }
    
    # Generate prices based on category
    price_base = {
        'Budget': 300,
        'Standard': 800,
        'Premium': 1500,
    }
    
    prices = []
    for cat in data['category']:
        base = price_base[cat]
        variation = np.random.normal(0, base * 0.1)  # 10% standard deviation
        price = max(base + variation, base * 0.7)  # Ensure no negative prices
        prices.append(round(price, 2))
    
    data['price'] = prices
    
    # Generate quantity sold with seasonal patterns
    quantity = []
    for date in selected_dates:
        # Higher sales in summer months (June-August)
        month = date.month
        if 6 <= month <= 8:
            base_quantity = np.random.poisson(5)
        else:
            base_quantity = np.random.poisson(3)
        
        # Weekend boost
        if date.dayofweek >= 5:  # Saturday or Sunday
            base_quantity += np.random.poisson(2)
            
        quantity.append(max(1, base_quantity))  # At least 1 sold
    
    data['quantity_sold'] = quantity
    
    # Calculate revenue
    data['revenue'] = [p * q for p, q in zip(data['price'], data['quantity_sold'])]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some special discount events
    special_dates = ['2023-07-04', '2023-11-24', '2023-12-26']
    for special_date in special_dates:
        idx = df[df['date'] == pd.to_datetime(special_date)].index
        df.loc[idx, 'price'] = df.loc[idx, 'price'] * 0.8
        df.loc[idx, 'quantity_sold'] = df.loc[idx, 'quantity_sold'] * 2
        df.loc[idx, 'revenue'] = df.loc[idx, 'price'] * df.loc[idx, 'quantity_sold']
    
    return df

# ---------------------------
# Streamlit App Configuration
# ---------------------------

st.set_page_config(
    page_title=TITLE,
    page_icon="📊",
)
st.title(TITLE)

# ---------------------------
# OpenAI API Key Entry and Test
# ---------------------------

st.sidebar.header("Enter your OpenAI API Key")

st.session_state["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Get API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Test OpenAI API Key
if openai_api_key:
    # Set the API key for OpenAI
    client = OpenAI(api_key=openai_api_key)
    
    # Wrap with LangSmith for observability
    if LANGSMITH_ENABLED:
        client = wrap_openai(client)

    # Test the API key
    try:
        # Fetch models to validate the key
        models = client.models.list()
        st.success("API Key is valid!")
    except Exception as e:
        st.error(f"Invalid API Key: {e}")
else:
    st.error("OpenAI API Key not found in environment variables.")
    st.stop()


# * OpenAI Model Selection

model_option = st.sidebar.selectbox("Choose OpenAI model", MODEL_LIST, index=0)

# Initialize the LLM with tracing
llm = ChatOpenAI(model=model_option, api_key=openai_api_key)

# Add observability UI elements to sidebar
if LANGSMITH_ENABLED:
    st.sidebar.subheader("LangSmith Observability")
    
    # Toggle tracing on/off
    enable_tracing = st.sidebar.checkbox("Enable detailed tracing", value=True)
    if enable_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        st.sidebar.success("LangSmith tracing enabled")
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        st.sidebar.warning("LangSmith tracing disabled")
    
    # Add expander with info
    with st.sidebar.expander("LangSmith Tracing Info"):
        st.write("""
        LangSmith tracing provides detailed logs of:
        - LLM calls and tokens usage
        - Data transformations
        - Agent operations and decisions
        - Generated code and visualizations
        
        Each query generates a unique Run ID that can be viewed in the LangSmith UI.
        """)
    
    # Show current run ID
    run_id = st.sidebar.empty()
    trace_link = st.sidebar.empty()
    
    # Add link to LangSmith dashboard
    st.sidebar.markdown("[Open LangSmith Dashboard](https://smith.langchain.com)")
else:
    st.sidebar.warning("LangSmith tracing disabled")

# ---------------------------
# Application Mode Selection
# ---------------------------

app_mode = st.sidebar.selectbox("Wybierz tryb aplikacji", APP_MODES)

# ---------------------------
# File Upload and Data Preview
# ---------------------------

st.markdown("""
Wgraj plik CSV lub Excel z danymi do analizy lub użyj przykładowych danych.  
""")

# Add file tracking to detect changes
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

use_sample_data = st.checkbox("Użyj przykładowych danych o sprzedaży rowerów", value=False)

if use_sample_data:
    df = create_sample_bike_sales_data()
    
    # Check if data source has changed
    if st.session_state.current_file_name != "sample_data":
        # Reset session state for new data
        st.session_state.data_history = {
            "raw": None,
            "cleaned": None, 
            "engineered": None,
            "current": None,
            "cleaning_explanation": None,
            "engineering_explanation": None,
        }
        st.session_state.config_analysis_results = None
        st.session_state.config_analyzed = False
        st.session_state.data_chat_history = []
        st.session_state.plots = []
        st.session_state.dataframes = []
        if "langchain_messages" in st.session_state:
            st.session_state.langchain_messages = []
        st.session_state.current_file_name = "sample_data"
    
    st.success("Załadowano przykładowe dane o sprzedaży rowerów.")
    
    st.subheader("Data Preview")
    st.dataframe(df.head())
else:
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file", type=["csv", "xlsx", "xls"]
    )
    if uploaded_file is not None:
        # Check if a new file has been uploaded
        if st.session_state.current_file_name != uploaded_file.name:
            # Reset session state for new data
            st.session_state.data_history = {
                "raw": None,
                "cleaned": None, 
                "engineered": None,
                "current": None,
                "cleaning_explanation": None,
                "engineering_explanation": None,
            }
            st.session_state.config_analysis_results = None
            st.session_state.config_analyzed = False
            st.session_state.data_chat_history = []
            st.session_state.plots = []
            st.session_state.dataframes = []
            if "langchain_messages" in st.session_state:
                st.session_state.langchain_messages = []
            # Ensure we reset any other relevant session state variables
            if "is_followup" in st.session_state:
                st.session_state.is_followup = False
            st.session_state.current_file_name = uploaded_file.name
        
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Data Preview")
        st.dataframe(df.head())
    else:
        st.info("Please upload a CSV or Excel file or use the sample data option to get started.")
        st.stop()

# ---------------------------
# Initialize Storage for Session State
# ---------------------------

if "plots" not in st.session_state:
    st.session_state.plots = []

if "dataframes" not in st.session_state:
    st.session_state.dataframes = []

# Store data processing history to enable follow-up questions
if "data_history" not in st.session_state:
    st.session_state.data_history = {
        "raw": None,
        "cleaned": None, 
        "engineered": None,
        "current": None,
        "cleaning_explanation": None,
        "engineering_explanation": None,
    }

# Store configuration analysis results
if "config_analysis_results" not in st.session_state:
    st.session_state.config_analysis_results = None

# Flag to track if we're processing a follow-up question
if "is_followup" not in st.session_state:
    st.session_state.is_followup = False

# Flag to track if configuration analysis has been performed
if "config_analyzed" not in st.session_state:
    st.session_state.config_analyzed = False

# Store chat history for data chat
if "data_chat_history" not in st.session_state:
    st.session_state.data_chat_history = []

# ---------------------------
# AI Agent Setup
# ---------------------------

LOG = False

# Instantiate DataCleaningAgent
data_cleaning_agent = DataCleaningAgent(
    model=llm,
    log=LOG,
    bypass_recommended_steps=True,
    n_samples=100,
)

# Instantiate FeatureEngineeringAgent
feature_agent = FeatureEngineeringAgent(
    model=llm,
    log=LOG,
    bypass_recommended_steps=True,
    n_samples=100,
)

# Instantiate DataVisualizationAgent
data_visualization_agent = DataVisualizationAgent(
    model=llm,
    n_samples=100,
    log=LOG,
)

# Instantiate PandasDataAnalyst
pandas_data_analyst = PandasDataAnalyst(
    model=llm,
    data_wrangling_agent=DataWranglingAgent(
        model=llm,
        log=LOG,
        bypass_recommended_steps=True,
        n_samples=100,
    ),
    data_visualization_agent=data_visualization_agent,
)

# Instantiate ConfigurationAnalysisAgent
config_analysis_agent = ConfigurationAnalysisAgent(
    model=llm,
    data_cleaning_agent=data_cleaning_agent,
    data_visualization_agent=data_visualization_agent,
    n_samples=100,
    log=LOG,
)

# Instantiate DataChatAgent
data_chat_agent = DataChatAgent(
    model=llm,
    n_samples=100,
    log=LOG,
)

# ---------------------------
# Chat with Data Mode Functions
# ---------------------------

def display_chat_with_data():
    """Display the chat interface for interacting with data."""
    st.subheader("Chat with your Data")
    
    # Initialize chat history from session state if available
    chat_history = st.session_state.data_chat_history
    
    # Display chat history
    for msg in chat_history:
        with st.chat_message("human" if "user" in msg else "assistant"):
            if "user" in msg:
                st.write(msg["user"])
            else:
                st.write(msg["assistant"]["answer"])
                
                # Display follow-up suggestions if available
                if "suggested_followups" in msg["assistant"]:
                    followups = msg["assistant"]["suggested_followups"]
                    if followups and isinstance(followups, list) and len(followups) > 0:
                        with st.expander("Suggested follow-up questions"):
                            for q in followups:
                                st.markdown(f"- {q.strip()}")
    
    # Chat input
    if user_query := st.chat_input("Ask a question about your data:"):
        # Display user message
        with st.chat_message("human"):
            st.write(user_query)
        
        # Store the original data if this is first question
        if st.session_state.data_history["raw"] is None:
            st.session_state.data_history["raw"] = df.copy()
            st.session_state.data_history["current"] = df.copy()
        
        # Process the query with a spinner
        with st.spinner("Analyzing your question..."):
            # Get current data state
            current_data = st.session_state.data_history["current"]
            
            # Get analysis results if available
            analysis_results = st.session_state.config_analysis_results if st.session_state.config_analyzed else {}
            
            # Invoke the data chat agent
            data_chat_agent.invoke_agent(
                data_raw=df,
                data_cleaned=current_data,
                analysis_results=analysis_results,
                user_query=user_query
            )
            
            # Get the response
            response = data_chat_agent.get_response()
            
            # Update chat history in session state
            chat_entry = {
                "user": user_query,
                "assistant": response
            }
            st.session_state.data_chat_history.append(chat_entry)
            
            # Display the response
            with st.chat_message("assistant"):
                st.write(response["answer"])
                
                # Display follow-up suggestions
                if "suggested_followups" in response:
                    followups = response["suggested_followups"]
                    if followups and isinstance(followups, list) and len(followups) > 0:
                        with st.expander("Suggested follow-up questions"):
                            for q in followups:
                                st.markdown(f"- {q.strip()}")

# ---------------------------
# Configuration Analysis Mode Functions
# ---------------------------

def display_config_analysis():
    """Display the configuration-based analysis interface."""
    st.subheader("Analyze Data with Configuration File")
    
    # Add option to download sample configuration
    sample_config_path = os.path.join(os.path.dirname(__file__), "sample_config.txt")
    if os.path.exists(sample_config_path):
        with open(sample_config_path, 'r') as f:
            sample_config = f.read()
        
        col1, col2 = st.columns([3, 1])
        with col2:
            st.download_button(
                label="Download Sample Config",
                data=sample_config,
                file_name="sample_config.txt",
                mime="text/plain"
            )
        with col1:
            st.info("You can download a sample configuration file to use as a template.")
    
    # Allow user to enter or upload a configuration file
    config_option = st.radio(
        "Choose configuration source",
        ["Enter configuration text", "Upload configuration file", "Use sample configuration"]
    )
    
    config_content = None
    config_file_path = None
    
    if config_option == "Enter configuration text":
        config_text = st.text_area(
            "Enter configuration text:",
            height=300,
            placeholder="""Ogólny opis danych:
- Zbiór danych zawiera informacje o sprzedaży rowerów.
- Dostępne kolumny: { df.columns.tolist() }
Analiza danych:
Page 1:
    - Title: "Raport z analizy danych"
    - Charts: "Histogram z rozkładu cen rowerów"
    - Description: "Opis rozkładu cen rowerów. Wyciąganie wniosków dotyczących rozkładu cen rowerów."
..."""
        )
        if config_text:
            # Write to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
                temp_file.write(config_text)
                config_file_path = temp_file.name
    elif config_option == "Upload configuration file":
        config_file = st.file_uploader(
            "Upload configuration file (YAML, JSON, or TXT)",
            type=["yaml", "yml", "json", "txt"],
            key="config_upload"
        )
        if config_file:
            # Write to a temporary file
            file_extension = config_file.name.split('.')[-1].lower()
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=f'.{file_extension}') as temp_file:
                temp_file.write(config_file.getvalue())
                config_file_path = temp_file.name
    else:  # Use sample configuration
        # Load the sample configuration
        if os.path.exists(sample_config_path):
            st.code(sample_config, language="yaml")
            st.info("This sample configuration will be used to analyze the data.")
            
            # Write to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
                temp_file.write(sample_config)
                config_file_path = temp_file.name
        else:
            st.error("Sample configuration file not found. Please choose another option.")
    
    # Run analysis button
    if config_file_path and st.button("Run Analysis"):
        with st.spinner("Running configuration-based analysis..."):
            # Store the original data if this is first question
            if st.session_state.data_history["raw"] is None:
                st.session_state.data_history["raw"] = df.copy()
                st.session_state.data_history["current"] = df.copy()
            
            # Run the configuration analysis
            config_analysis_agent.invoke_agent(
                data_raw=df,
                config_file_path=config_file_path
            )
            
            # Store results in session state
            st.session_state.config_analysis_results = config_analysis_agent.get_analysis_results()
            st.session_state.config_analyzed = True
            
            # Get cleaned data if it exists
            cleaned_data = config_analysis_agent.get_data_cleaned()
            if cleaned_data is not None:
                st.session_state.data_history["cleaned"] = cleaned_data
                st.session_state.data_history["current"] = cleaned_data
            
            # Display success message
            st.success("Analysis completed successfully!")
    
    # Display results if analysis has been run
    if st.session_state.config_analyzed and st.session_state.config_analysis_results:
        st.subheader("Analysis Results")
        
        # Get total pages
        total_pages = config_analysis_agent.get_total_pages()
        
        # Create tabs for each page
        if total_pages > 0:
            tabs = st.tabs([f"Page {i+1}" for i in range(total_pages)])
            
            for i, tab in enumerate(tabs):
                with tab:
                    page_key = f"Page {i+1}"
                    if page_key in st.session_state.config_analysis_results:
                        page_results = st.session_state.config_analysis_results[page_key]
                        
                        # Display title
                        st.markdown(f"## {page_results.get('title', f'Analysis Page {i+1}')}")
                        
                        # Display chart description
                        st.markdown(f"**Chart Description:** {page_results.get('charts_description', '')}")
                        
                        # Display visualization if available
                        visualizations = config_analysis_agent.get_visualizations()
                        if page_key in visualizations:
                            st.plotly_chart(visualizations[page_key])
                        else:
                            # If no visualization available, show the visualization code
                            with st.expander("View visualization code"):
                                st.code(page_results.get('visualization_code', ''), language="python")
                        
                        # Display analysis
                        st.markdown("### Analysis")
                        st.markdown(page_results.get('analysis_description', ''))
        else:
            st.warning("No analysis pages were found in the configuration.")

# ---------------------------
# Standard Chat Interface for Data Analysis
# ---------------------------

def display_chat_history():
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            if "PLOT_INDEX:" in msg.content:
                plot_index = int(msg.content.split("PLOT_INDEX:")[1])
                st.plotly_chart(
                    st.session_state.plots[plot_index], key=f"history_plot_{plot_index}"
                )
            elif "DATAFRAME_INDEX:" in msg.content:
                df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                st.dataframe(
                    st.session_state.dataframes[df_index],
                    key=f"history_dataframe_{df_index}",
                )
            else:
                st.write(msg.content)

def process_standard_chat(question):
    """Process a query in the standard data analysis chat mode."""
    if not st.session_state["OPENAI_API_KEY"]:
        st.error("Please enter your OpenAI API Key to proceed.")
        st.stop()

    # Generate a run ID for tracking
    if LANGSMITH_ENABLED:
        current_run_id = str(uuid.uuid4())
        st.session_state["run_id"] = current_run_id
        
        # Update sidebar with run ID
        run_id.text(f"Run ID: {current_run_id}")
        trace_link.markdown(f"[View trace in LangSmith](https://smith.langchain.com/o/viewer/traces/{current_run_id})")
        
        # Log the start of query processing
        log_action("query_started", 
                   metadata={"question": question, "file": st.session_state.current_file_name},
                   run_id=current_run_id)

    with st.spinner("Thinking..."):
        st.chat_message("human").write(question)
        msgs.add_user_message(question)

        # Store original data if this is first question
        if st.session_state.data_history["raw"] is None:
            st.session_state.data_history["raw"] = df.copy()
            st.session_state.data_history["current"] = df.copy()
            
            if LANGSMITH_ENABLED:
                # Log data loaded event
                log_action("data_loaded", 
                           metadata={"shape": df.shape, "columns": df.columns.tolist()},
                           run_id=current_run_id)

        # Check if this is a follow-up question about previous processing or data
        followup_analysis_prompt = f"""
        Analyze if this user query is a follow-up question about previous data processing results or asking for a targeted correction:
        User query: "{question}"
        
        Determine if the query:
        1. Is asking about why specific data was processed in a certain way (explanation request)
        2. Is asking to correct or modify specific data elements from previous processing (correction request)
        3. Is a new standalone question (new request)
        
        Return a JSON with these keys:
        - "request_type": one of ["explanation", "correction", "new"]
        - "focuses_on": one of ["cleaning", "engineering", "analysis", "general"] (which step it's asking about)
        - "data_elements": list of specific data elements mentioned (column names, values, etc.)
        
        Example: {{"request_type": "explanation", "focuses_on": "cleaning", "data_elements": ["outliers", "price"]}}
        
        IMPORTANT: Return ONLY the JSON object with no additional text.
        """
        
        try:
            followup_response = llm.invoke(followup_analysis_prompt)
            response_content = followup_response.content.strip()
            
            # Parse JSON response with error handling
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                response_content = json_match.group(0)
            
            if response_content.startswith("```json"):
                response_content = response_content.replace("```json", "", 1).replace("```", "", 1)
            elif response_content.startswith("```"):
                response_content = response_content.replace("```", "", 1).replace("```", "", 1)
                
            followup_decision = json.loads(response_content.strip())
            request_type = followup_decision.get("request_type", "new")
            focuses_on = followup_decision.get("focuses_on", "general")
            data_elements = followup_decision.get("data_elements", [])
            
            # Set flag for followup processing
            st.session_state.is_followup = request_type in ["explanation", "correction"]
            
        except Exception as e:
            # Default to new question if parsing fails
            request_type = "new"
            focuses_on = "general"
            data_elements = []
            st.session_state.is_followup = False
            
        # Handle follow-up explanation requests
        if request_type == "explanation":
            if focuses_on == "cleaning" and st.session_state.data_history["cleaning_explanation"]:
                explanation = st.session_state.data_history["cleaning_explanation"]
                
                # Get more specific explanation if data elements are mentioned
                if data_elements:
                    specific_prompt = f"""
                    The user is asking about the cleaning process for these specific elements: {', '.join(data_elements)}
                    
                    Here is the full explanation of the cleaning process:
                    {explanation}
                    
                    Provide a targeted explanation focusing ONLY on how those specific elements were processed.
                    """
                    specific_explanation = llm.invoke(specific_prompt).content
                    st.chat_message("ai").write(specific_explanation)
                    msgs.add_ai_message(specific_explanation)
                else:
                    st.chat_message("ai").write(explanation)
                    msgs.add_ai_message(explanation)
                
            elif focuses_on == "engineering" and st.session_state.data_history["engineering_explanation"]:
                explanation = st.session_state.data_history["engineering_explanation"]
                
                # Get more specific explanation if data elements are mentioned
                if data_elements:
                    specific_prompt = f"""
                    The user is asking about the feature engineering process for these specific elements: {', '.join(data_elements)}
                    
                    Here is the full explanation of the engineering process:
                    {explanation}
                    
                    Provide a targeted explanation focusing ONLY on how those specific elements were processed.
                    """
                    specific_explanation = llm.invoke(specific_prompt).content
                    st.chat_message("ai").write(specific_explanation)
                    msgs.add_ai_message(specific_explanation)
                else:
                    st.chat_message("ai").write(explanation)
                    msgs.add_ai_message(explanation)
            else:
                # No relevant explanation found
                st.chat_message("ai").write("I don't have a detailed explanation for that specific processing step.")
                msgs.add_ai_message("I don't have a detailed explanation for that specific processing step.")
                
            # Skip the rest of processing for explanation requests
            st.stop()
            
        # Handle follow-up correction requests
        if request_type == "correction":
            current_df = st.session_state.data_history["current"]
            
            if focuses_on == "cleaning":
                # Go back to raw data for corrections related to cleaning
                base_df = st.session_state.data_history["raw"]
                correction_agent = data_cleaning_agent
                
                # Create targeted cleaning instructions
                targeted_instructions = f"""
                This is a targeted correction request. The user wants to fix specific issues with: {', '.join(data_elements)}
                
                Only apply cleaning operations to these specific elements. Do not modify other parts of the data.
                
                User's correction request: {question}
                """
                
                st.chat_message("ai").write("Applying targeted cleaning correction...")
                msgs.add_ai_message("Applying targeted cleaning correction...")
                
                try:
                    correction_agent.invoke_agent(
                        user_instructions=targeted_instructions,
                        data_raw=base_df,
                    )
                    corrected_df = correction_agent.get_data_cleaned()
                    
                    if corrected_df is not None:
                        # Store explanation for future reference
                        if hasattr(correction_agent, "get_workflow_summary"):
                            explanation = correction_agent.get_workflow_summary() or "Targeted cleaning was performed."
                        else:
                            explanation = "Targeted cleaning was performed based on your request."
                        
                        st.session_state.data_history["cleaned"] = corrected_df
                        st.session_state.data_history["current"] = corrected_df
                        st.session_state.data_history["cleaning_explanation"] = explanation
                        
                        st.chat_message("ai").write("Targeted cleaning correction applied successfully.")
                        msgs.add_ai_message("Targeted cleaning correction applied successfully.")
                        
                        # Display the corrected data
                        df_index = len(st.session_state.dataframes)
                        st.session_state.dataframes.append(corrected_df)
                        msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                        st.dataframe(corrected_df)
                    else:
                        st.chat_message("ai").write("The targeted cleaning did not result in any changes to the data.")
                        msgs.add_ai_message("The targeted cleaning did not result in any changes to the data.")
                except Exception as e:
                    error_msg = f"Error applying targeted correction: {str(e)}"
                    st.chat_message("ai").write(error_msg)
                    msgs.add_ai_message(error_msg)
                
                # Skip the rest of processing for correction requests
                st.stop()
                
            elif focuses_on == "engineering":
                # Use cleaned data as base for engineering corrections if available
                base_df = st.session_state.data_history["cleaned"] if st.session_state.data_history["cleaned"] is not None else current_df
                correction_agent = feature_agent
                
                # Create targeted engineering instructions
                targeted_instructions = f"""
                This is a targeted correction request. The user wants to fix specific engineered features: {', '.join(data_elements)}
                
                Only apply feature engineering operations to these specific elements. Do not modify other parts of the data.
                
                User's correction request: {question}
                """
                
                st.chat_message("ai").write("Applying targeted feature engineering correction...")
                msgs.add_ai_message("Applying targeted feature engineering correction...")
                
                try:
                    correction_agent.invoke_agent(
                        user_instructions=targeted_instructions,
                        data_raw=base_df,
                    )
                    corrected_df = correction_agent.get_data_engineered()
                    
                    if corrected_df is not None:
                        # Store explanation for future reference
                        if hasattr(correction_agent, "get_workflow_summary"):
                            explanation = correction_agent.get_workflow_summary() or "Targeted feature engineering was performed."
                        else:
                            explanation = "Targeted feature engineering was performed based on your request."
                        
                        st.session_state.data_history["engineered"] = corrected_df
                        st.session_state.data_history["current"] = corrected_df
                        st.session_state.data_history["engineering_explanation"] = explanation
                        
                        st.chat_message("ai").write("Targeted feature engineering correction applied successfully.")
                        msgs.add_ai_message("Targeted feature engineering correction applied successfully.")
                        
                        # Display the corrected data
                        df_index = len(st.session_state.dataframes)
                        st.session_state.dataframes.append(corrected_df)
                        msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                        st.dataframe(corrected_df)
                    else:
                        st.chat_message("ai").write("The targeted feature engineering did not result in any changes to the data.")
                        msgs.add_ai_message("The targeted feature engineering did not result in any changes to the data.")
                except Exception as e:
                    error_msg = f"Error applying targeted correction: {str(e)}"
                    st.chat_message("ai").write(error_msg)
                    msgs.add_ai_message(error_msg)
                
                # Skip the rest of processing for correction requests
                st.stop()

        # For new questions, continue with the normal processing flow
        current_df = st.session_state.data_history["current"]

        # --- Workflow Analysis Step ---
        # Use the LLM to analyze the user query and determine which data processing steps are needed
        workflow_analysis_prompt = f"""
        Analyze the following user query for a data analysis task and determine which steps are needed.
        Query: "{question}"
        
        Determine if the query suggests a need for:
        1. Data cleaning (e.g., handling missing values, outliers, duplicates)
        2. Feature engineering (e.g., creating new columns, transforming data, encoding)
        
        Return a JSON with two boolean keys: "needs_cleaning" and "needs_feature_engineering"
        Example: {{"needs_cleaning": true, "needs_feature_engineering": false}}
        
        IMPORTANT: Return ONLY the JSON object with no additional text, explanations, or Markdown formatting.
        """
        
        try:
            workflow_response = llm.invoke(workflow_analysis_prompt)
            response_content = workflow_response.content.strip()
            
            # Handle potential markdown code block formatting
            if response_content.startswith("```json"):
                response_content = response_content.replace("```json", "", 1)
                response_content = response_content.replace("```", "", 1)
            elif response_content.startswith("```"):
                response_content = response_content.replace("```", "", 1)
                response_content = response_content.replace("```", "", 1)
            
            # Find JSON object if wrapped in text
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                response_content = json_match.group(0)
                
            workflow_decision = json.loads(response_content.strip())
            needs_cleaning = workflow_decision.get("needs_cleaning", False)
            needs_feature_engineering = workflow_decision.get("needs_feature_engineering", False)
            
            # Log the decision to the user
            decision_msg = f"Analysis: Your query {'requires' if needs_cleaning else 'does not require'} data cleaning and {'requires' if needs_feature_engineering else 'does not require'} feature engineering."
            st.chat_message("ai").write(decision_msg)
            msgs.add_ai_message(decision_msg)
            
        except Exception as e:
            # If the workflow analysis fails, assume we need both steps to be safe
            needs_cleaning = True
            needs_feature_engineering = True
            st.chat_message("ai").write(f"Analyzing your request... (Using default workflow)")
            msgs.add_ai_message("Analyzing your request... (Using default workflow)")

        # --- Data Cleaning Step (Conditional) ---
        if needs_cleaning:
            try:
                st.chat_message("ai").write("Cleaning data...")
                msgs.add_ai_message("Cleaning data...")
                
                if LANGSMITH_ENABLED:
                    log_action("data_cleaning_started", run_id=current_run_id)
                
                data_cleaning_agent.invoke_agent(
                    user_instructions=question,
                    data_raw=current_df,
                )
                cleaned_df = data_cleaning_agent.get_data_cleaned()
                
                if cleaned_df is not None:
                    # Store explanation for future reference
                    if hasattr(data_cleaning_agent, "get_workflow_summary"):
                        explanation = data_cleaning_agent.get_workflow_summary() or "Data cleaning was performed."
                    else:
                        explanation = "Data cleaning was performed based on general best practices and your request."
                    
                    st.session_state.data_history["cleaned"] = cleaned_df
                    st.session_state.data_history["current"] = cleaned_df
                    st.session_state.data_history["cleaning_explanation"] = explanation
                    
                    if LANGSMITH_ENABLED:
                        # Log cleaning results
                        log_action("data_cleaning_completed", 
                                  metadata={
                                      "cleaned_shape": cleaned_df.shape,
                                      "diff_rows": current_df.shape[0] - cleaned_df.shape[0],
                                      "explanation": explanation[:500]  # Truncate if too long
                                  },
                                  run_id=current_run_id)
                    
                    st.chat_message("ai").write("Data cleaning complete. Using cleaned data.")
                    msgs.add_ai_message("Data cleaning complete. Using cleaned data.")
                    current_df = cleaned_df
                else:
                    st.chat_message("ai").write("Data cleaning did not produce new data. Using original data.")
                    msgs.add_ai_message("Data cleaning did not produce new data. Using original data.")
            except Exception as dc_error:
                error_msg = f"An error occurred during data cleaning. Using original data for analysis."
                st.chat_message("ai").write(error_msg)
                msgs.add_ai_message(error_msg)
                
                if LANGSMITH_ENABLED:
                    log_action("data_cleaning_error", 
                              metadata={"error": str(dc_error)},
                              run_id=current_run_id)
        else:
            st.chat_message("ai").write("Skipping data cleaning as it doesn't appear necessary for your query.")
            msgs.add_ai_message("Skipping data cleaning as it doesn't appear necessary for your query.")

        # --- Feature Engineering Step (Conditional) ---
        if needs_feature_engineering:
            try:
                st.chat_message("ai").write("Performing feature engineering...")
                msgs.add_ai_message("Performing feature engineering...")
                
                if LANGSMITH_ENABLED:
                    log_action("feature_engineering_started", run_id=current_run_id)
                
                feature_agent.invoke_agent(
                    user_instructions=question,
                    data_raw=current_df,
                )
                engineered_df = feature_agent.get_data_engineered()
                if engineered_df is not None:
                    # Store explanation for future reference
                    if hasattr(feature_agent, "get_workflow_summary"):
                        explanation = feature_agent.get_workflow_summary() or "Feature engineering was performed."
                    else:
                        explanation = "Feature engineering was performed based on your request."
                    
                    st.session_state.data_history["engineered"] = engineered_df
                    st.session_state.data_history["current"] = engineered_df
                    st.session_state.data_history["engineering_explanation"] = explanation
                    
                    if LANGSMITH_ENABLED:
                        # Log feature engineering results
                        new_columns = set(engineered_df.columns) - set(current_df.columns)
                        log_action("feature_engineering_completed", 
                                  metadata={
                                      "engineered_shape": engineered_df.shape,
                                      "new_columns": list(new_columns),
                                      "explanation": explanation[:500]  # Truncate if too long
                                  },
                                  run_id=current_run_id)
                    
                    st.chat_message("ai").write("Feature engineering complete. Using engineered data.")
                    msgs.add_ai_message("Feature engineering complete. Using engineered data.")
                    current_df = engineered_df
                else:
                    st.chat_message("ai").write("Feature engineering did not produce new data. Using current data.")
                    msgs.add_ai_message("Feature engineering did not produce new data. Using current data.")
            except Exception as fe_error:
                error_msg = f"An error occurred during feature engineering. Using current data for analysis."
                st.chat_message("ai").write(error_msg)
                msgs.add_ai_message(error_msg)
                
                if LANGSMITH_ENABLED:
                    log_action("feature_engineering_error", 
                              metadata={"error": str(fe_error)},
                              run_id=current_run_id)
        else:
            st.chat_message("ai").write("Skipping feature engineering as it doesn't appear necessary for your query.")
            msgs.add_ai_message("Skipping feature engineering as it doesn't appear necessary for your query.")

        try:
            if LANGSMITH_ENABLED:
                log_action("analysis_started", run_id=current_run_id)
                
            pandas_data_analyst.invoke_agent(
                user_instructions=question,
                data_raw=current_df,
            )
            result = pandas_data_analyst.get_response()
            
            if LANGSMITH_ENABLED:
                # Log analysis results
                log_action("analysis_completed", 
                          metadata={
                              "routing": result.get("routing_preprocessor_decision", "unknown"),
                              "has_plot": result.get("plotly_graph") is not None,
                              "has_table": result.get("data_wrangled") is not None
                          },
                          run_id=current_run_id)
        except Exception as e:
            st.chat_message("ai").write(
                "An error occurred while processing your query. Please try again."
            )
            msgs.add_ai_message(
                "An error occurred while processing your query. Please try again."
            )
            
            if LANGSMITH_ENABLED:
                log_action("analysis_error", 
                          metadata={"error": str(e)},
                          run_id=current_run_id)
            
            st.stop()

        routing = result.get("routing_preprocessor_decision")

        if routing == "chart" and not result.get("plotly_error", False):
            # Process chart result
            plot_data = result.get("plotly_graph")
            if plot_data:
                # Convert dictionary to JSON string if needed
                if isinstance(plot_data, dict):
                    plot_json = json.dumps(plot_data)
                else:
                    plot_json = plot_data
                plot_obj = pio.from_json(plot_json)
                response_text = "Returning the generated chart."
                # Store the chart
                plot_index = len(st.session_state.plots)
                st.session_state.plots.append(plot_obj)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"PLOT_INDEX:{plot_index}")
                st.chat_message("ai").write(response_text)
                st.plotly_chart(plot_obj)
                
                # Add feedback collection if LangSmith is enabled
                if LANGSMITH_ENABLED and "run_id" in st.session_state:
                    current_run = st.session_state["run_id"]
                    col1, col2 = st.columns([1, 8])
                    with col1:
                        if st.button("👍", key=f"thumbs_up_{current_run}"):
                            try:
                                ls_client.create_feedback(
                                    current_run,
                                    key="user-score",
                                    score=1.0,
                                    comment="User liked the result"
                                )
                                st.success("Feedback recorded!")
                            except Exception as e:
                                st.error(f"Failed to record feedback: {e}")
                    with col2:
                        if st.button("👎", key=f"thumbs_down_{current_run}"):
                            try:
                                ls_client.create_feedback(
                                    current_run,
                                    key="user-score",
                                    score=0.0,
                                    comment="User disliked the result"
                                )
                                st.success("Feedback recorded!")
                            except Exception as e:
                                st.error(f"Failed to record feedback: {e}")
                
                if LANGSMITH_ENABLED:
                    # Log chart display
                    log_action("chart_displayed", 
                              metadata={
                                  "plot_type": result.get("plot_type", "unknown"),
                                  "plot_index": plot_index
                              },
                              run_id=current_run_id)
            else:
                st.chat_message("ai").write("The agent did not return a valid chart.")
                msgs.add_ai_message("The agent did not return a valid chart.")
                
                if LANGSMITH_ENABLED:
                    log_action("chart_error", run_id=current_run_id)

        elif routing == "table":
            # Process table result
            data_wrangled = result.get("data_wrangled")
            if data_wrangled is not None:
                response_text = "Returning the data table."
                # Ensure data_wrangled is a DataFrame
                if not isinstance(data_wrangled, pd.DataFrame):
                    data_wrangled = pd.DataFrame(data_wrangled)
                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(data_wrangled)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                st.chat_message("ai").write(response_text)
                st.dataframe(data_wrangled)
                
                if LANGSMITH_ENABLED:
                    # Log table display
                    log_action("table_displayed", 
                              metadata={
                                  "table_shape": data_wrangled.shape,
                                  "table_index": df_index,
                                  "columns": data_wrangled.columns.tolist()[:20]  # First 20 columns
                              },
                              run_id=current_run_id)
            else:
                st.chat_message("ai").write("No table data was returned by the agent.")
                msgs.add_ai_message("No table data was returned by the agent.")
                
                if LANGSMITH_ENABLED:
                    log_action("table_error", run_id=current_run_id)
        else:
            # Fallback if routing decision is unclear or if chart error occurred
            data_wrangled = result.get("data_wrangled")
            if data_wrangled is not None:
                response_text = (
                    "I apologize. There was an issue with generating the chart. "
                    "Returning the data table instead."
                )
                if not isinstance(data_wrangled, pd.DataFrame):
                    data_wrangled = pd.DataFrame(data_wrangled)
                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(data_wrangled)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                st.chat_message("ai").write(response_text)
                st.dataframe(data_wrangled)
                
                if LANGSMITH_ENABLED:
                    log_action("fallback_table_displayed", 
                              metadata={"table_index": df_index},
                              run_id=current_run_id)
            else:
                response_text = (
                    "An error occurred while processing your query. Please try again."
                )
                msgs.add_ai_message(response_text)
                st.chat_message("ai").write(response_text)
                
                if LANGSMITH_ENABLED:
                    log_action("response_error", run_id=current_run_id)
                    
        # Log completion of query processing
        if LANGSMITH_ENABLED:
            log_action("query_completed", run_id=current_run_id)

# ---------------------------
# Main Application Logic Based on Selected Mode
# ---------------------------

if app_mode == "Analiza danych z czatem":
    # Display example questions
    with st.expander("Example Questions", expanded=False):
        st.write(
            """
            Calculate the 7-day rolling average of 'quantity_sold' for each bike model. Then create appropiate chart presenting the data
            """
        )
    
    # Initialize chat history
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    
    # Clear messages if they are from a previous file/session
    if "file_changed_for_chat_mode" not in st.session_state:
        st.session_state.file_changed_for_chat_mode = True
        msgs.clear()
        msgs.add_ai_message("How can I help you?")
    elif st.session_state.file_changed_for_chat_mode:
        st.session_state.file_changed_for_chat_mode = False
    
    # Render current messages from StreamlitChatMessageHistory
    display_chat_history()
    
    # Chat input for standard data analysis mode
    if question := st.chat_input("Enter your question here:", key="query_input"):
        process_standard_chat(question)

elif app_mode == "Analiza z pliku konfiguracyjnego":
    # Display configuration-based analysis interface
    display_config_analysis()

elif app_mode == "Czatowanie z danymi":
    # Display chat with data interface
    display_chat_with_data()

# ---------------------------
# Utility Functions
# ---------------------------

def log_action(action, metadata=None, run_id=None):
    """Log an action to LangSmith for observability"""
    if not LANGSMITH_ENABLED:
        return None
    
    if run_id is None:
        run_id = str(uuid.uuid4())
    
    if metadata is None:
        metadata = {}
    
    # Add common metadata
    metadata.update({
        "action": action,
        "app_mode": app_mode,
        "model": model_option,
        "timestamp": str(datetime.now())
    })
    
    try:
        # Create run in LangSmith (can be used for standalone events)
        ls_client.create_run(
            name=f"User Action: {action}",
            run_type="llm",
            inputs={"action": action},
            outputs={"status": "completed"},
            runtime={"model": model_option},
            extra={
                "metadata": metadata
            },
            trace_id=run_id,
            run_id=run_id
        )
        
        # Update UI with run ID
        if "run_id" in st.session_state:
            run_id = st.sidebar.empty()
            run_id.text(f"Run ID: {run_id}")
            trace_link = st.sidebar.empty()
            trace_link.markdown(f"[View trace in LangSmith](https://smith.langchain.com/o/viewer/traces/{run_id})")
        
        return run_id
    except Exception as e:
        print(f"Failed to log action to LangSmith: {e}")
        return None
