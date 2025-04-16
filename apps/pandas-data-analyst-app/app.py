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

MODEL_LIST = ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
TITLE = "Analityk danych Sigla"

# Application modes
APP_MODES = ["Analiza i czatowanie z danymi", "Analiza z pliku konfiguracyjnego"]

# ---------------------------
# Streamlit App Configuration
# ---------------------------

st.set_page_config(
    page_title=TITLE,
    page_icon="📊",
)
st.title(TITLE)

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

# Store the last DataFrame generated by an analysis step for follow-up questions
if "last_generated_dataframe" not in st.session_state:
    st.session_state.last_generated_dataframe = None

# Store chat history for data chat
if "data_chat_history" not in st.session_state:
    st.session_state.data_chat_history = []

# ---------------------------
# AI Agent Setup
# ---------------------------

LOG = True

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
    sample_config_path = os.path.join(os.path.dirname(__file__), "sample_config.yaml")
    if os.path.exists(sample_config_path):
        with open(sample_config_path, 'r') as f:
            sample_config = f.read()
        
        col1, col2 = st.columns([3, 1])
        with col2:
            st.download_button(
                label="Download Sample Config",
                data=sample_config,
                file_name="sample_config.yamlt",
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
# Main Application Logic Based on Selected Mode
# ---------------------------

if app_mode == "Analiza i czatowanie z danymi":
    # Display example questions
    with st.expander("Example Questions", expanded=False):
        st.write(
            """
            Examples:
            - Calculate the 7-day rolling average of 'quantity_sold' for each bike model. Then create appropiate chart presenting the data
            - What is the data about?
            - Create a function that merges two columns
            - What's the relationship between price and quantity_sold?
            """
        )
    
    # Initialize unified chat history
    if "unified_chat_history" not in st.session_state:
        st.session_state.unified_chat_history = []
        
    # Display unified chat history
    for msg in st.session_state.unified_chat_history:
        if msg["type"] == "user":
            with st.chat_message("human"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                # Handle different response types
                if "text" in msg["content"]:
                    st.write(msg["content"]["text"])
                
                # Display any plot if available
                if "plot" in msg["content"] and msg["content"]["plot"] is not None:
                    st.plotly_chart(msg["content"]["plot"])
                
                # Display any dataframe if available
                if "dataframe" in msg["content"] and msg["content"]["dataframe"] is not None:
                    st.dataframe(msg["content"]["dataframe"])
                    
                # Display follow-up suggestions if available
                if "followups" in msg["content"] and msg["content"]["followups"]:
                    followups = msg["content"]["followups"]
                    if followups and isinstance(followups, list) and len(followups) > 0:
                        with st.expander("Suggested follow-up questions"):
                            for q in followups:
                                st.markdown(f"- {q.strip()}")
    
    # Unified chat input
    if user_query := st.chat_input("Ask about your data or request analysis:"):
        # Store the original data if this is first question
        if st.session_state.data_history["raw"] is None:
            st.session_state.data_history["raw"] = df.copy()
            st.session_state.data_history["current"] = df.copy()
        
        # Add user message to history
        st.session_state.unified_chat_history.append({
            "type": "user",
            "content": user_query
        })
        
        # Display user message
        with st.chat_message("human"):
            st.write(user_query)
            
        # Analyze query to determine the best agent to handle it
        with st.spinner("Analyzing your question..."):
            # Determine input data: use last generated table for follow-ups if requested
            is_followup_query = False
            input_data_for_agent = None
            # Simple check for follow-up phrases (can be improved with more robust NLP)
            followup_phrases = ["do tej tabeli", "do powyższej tabeli", "w tej tabeli", "na tej tabeli", "z tej tabeli"]
            if st.session_state.last_generated_dataframe is not None and any(phrase in user_query.lower() for phrase in followup_phrases):
                is_followup_query = True
                input_data_for_agent = st.session_state.last_generated_dataframe
                # Keep last_generated_dataframe for this query, it might be replaced by the result
            else:
                # Default to current data from history
                input_data_for_agent = st.session_state.data_history["current"]
                # If it's not a follow-up query targeting the last table, clear the context
                st.session_state.last_generated_dataframe = None

            # Get current data state (used primarily by chat agent or if no specific follow-up)
            current_data = st.session_state.data_history["current"] 
            
            # Query analysis to determine routing
            agent_routing_prompt = f"""
            Analyze this query and determine if it's better handled by:
            1. A general data chat agent (for simple questions about the data, explaining columns, etc.)
            2. A data analysis agent (for calculations, statistics, visualizations, data transformations)
            
            User query: "{user_query}"
            
            Return ONLY one of these values:
            - "chat" (for general data chat questions)
            - "analysis" (for data analysis operations)
            """
            
            # Determine which agent to use
            routing_response = llm.invoke(agent_routing_prompt).content.strip().lower()
            agent_type = "chat" if "chat" in routing_response else "analysis"
            
            # Process with appropriate agent
            if agent_type == "chat":
                # Chat agent usually operates on the main current data unless specifically asked otherwise
                # Clear any previous table context when switching to general chat
                st.session_state.last_generated_dataframe = None 
                
                # Get analysis results if available
                analysis_results = st.session_state.config_analysis_results if st.session_state.config_analyzed else {}
                
                # Process with DataChatAgent using the determined input data (or default if context cleared)
                data_chat_agent.invoke_agent(
                    data_raw=df, # Keep raw as original df
                    data_cleaned=input_data_for_agent if input_data_for_agent is not None else current_data, # Use selected data, fallback to current
                    analysis_results=analysis_results,
                    user_query=user_query
                )
                
                # Get the response
                chat_response = data_chat_agent.get_response()
                
                # Format response for unified history
                response_content = {
                    "text": chat_response["answer"],
                    "followups": chat_response.get("suggested_followups", [])
                }
                
                # Add to unified history
                st.session_state.unified_chat_history.append({
                    "type": "assistant",
                    "content": response_content
                })
                
                # Display the response
                with st.chat_message("assistant"):
                    st.write(response_content["text"])
                    
                    # Display follow-up suggestions
                    if response_content["followups"]:
                        with st.expander("Suggested follow-up questions"):
                            for q in response_content["followups"]:
                                st.markdown(f"- {q.strip()}")
            else:
                # Process with PandasDataAnalyst using the determined input data
                pandas_data_analyst.invoke_agent(
                    user_instructions=user_query,
                    data_raw=input_data_for_agent,
                )
                result = pandas_data_analyst.get_response()
                
                # Initialize response content
                response_content = {
                    "text": "Here are the analysis results:",
                    "plot": None,
                    "dataframe": None
                }
                
                # Process result based on routing decision
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
                        
                        # Update response content
                        response_content["text"] = "Here's the visualization based on your request."
                        response_content["plot"] = plot_obj
                
                elif routing == "table":
                    # Process table result
                    data_wrangled = result.get("data_wrangled")
                    if data_wrangled is not None:
                        # Ensure data_wrangled is a DataFrame
                        if not isinstance(data_wrangled, pd.DataFrame):
                            data_wrangled = pd.DataFrame(data_wrangled)
                            
                        # Update response content
                        response_content["text"] = "Here's the data table based on your request."
                        response_content["dataframe"] = data_wrangled
                        
                        # Store this dataframe as the new context for potential follow-ups
                        st.session_state.last_generated_dataframe = data_wrangled.copy()
                
                # Add to unified history
                st.session_state.unified_chat_history.append({
                    "type": "assistant",
                    "content": response_content
                })
                
                # Display the response
                with st.chat_message("assistant"):
                    st.write(response_content["text"])
                    
                    if response_content["plot"] is not None:
                        st.plotly_chart(response_content["plot"])
                        
                    if response_content["dataframe"] is not None:
                        st.dataframe(response_content["dataframe"])

elif app_mode == "Analiza z pliku konfiguracyjnego":
    # Display configuration-based analysis interface
    display_config_analysis()
