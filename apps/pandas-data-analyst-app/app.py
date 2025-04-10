# BUSINESS SCIENCE
# Pandas Data Analyst App
# -----------------------

# This app is designed to help you analyze data and create data visualizations from natural language requests.

# Imports
# !pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from openai import OpenAI
import os
import streamlit as st
import pandas as pd
import plotly.io as pio
import json
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

from ai_data_science_team import (
    PandasDataAnalyst,
    DataWranglingAgent,
    DataVisualizationAgent,
    FeatureEngineeringAgent,
    DataCleaningAgent,
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

# * APP INPUTS ----

MODEL_LIST = ["gpt-4o-mini", "gpt-4o"]
TITLE = "Analityk danych Sigla"

# ---------------------------
# Streamlit App Configuration
# ---------------------------

st.set_page_config(
    page_title=TITLE,
    page_icon="📊",
)
st.title(TITLE)


with st.expander("Example Questions", expanded=False):
    st.write(
        """
        # calculate the 7-day rolling average of 'quantity_sold' for each bike model. Then create appropiate chart presenting the data
        """
    )

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

llm = ChatOpenAI(model=model_option, api_key=openai_api_key)


# ---------------------------
# File Upload and Data Preview
# ---------------------------

st.markdown("""
Wgraj plik CSV i zadawaj pytania dotyczące danych.  
""")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file", type=["csv", "xlsx", "xls"]
)
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())
else:
    st.info("Please upload a CSV or Excel file to get started.")
    st.stop()

# ---------------------------
# Initialize Chat Message History and Storage
# ---------------------------

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

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

# Flag to track if we're processing a follow-up question
if "is_followup" not in st.session_state:
    st.session_state.is_followup = False

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


# Render current messages from StreamlitChatMessageHistory
display_chat_history()

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

pandas_data_analyst = PandasDataAnalyst(
    model=llm,
    data_wrangling_agent=DataWranglingAgent(
        model=llm,
        log=LOG,
        bypass_recommended_steps=True,
        n_samples=100,
    ),
    data_visualization_agent=DataVisualizationAgent(
        model=llm,
        n_samples=100,
        log=LOG,
    ),
)

# ---------------------------
# Chat Input and Agent Invocation
# ---------------------------

if question := st.chat_input("Enter your question here:", key="query_input"):
    if not st.session_state["OPENAI_API_KEY"]:
        st.error("Please enter your OpenAI API Key to proceed.")
        st.stop()

    with st.spinner("Thinking..."):
        st.chat_message("human").write(question)
        msgs.add_user_message(question)

        # Store original data if this is first question
        if st.session_state.data_history["raw"] is None:
            st.session_state.data_history["raw"] = df.copy()
            st.session_state.data_history["current"] = df.copy()

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
            st.chat_message("ai").write(f"Analyzing your request... (Using default workflow due to: {str(e)})")
            msgs.add_ai_message("Analyzing your request... (Using default workflow)")

        # --- Data Cleaning Step (Conditional) ---
        if needs_cleaning:
            try:
                st.chat_message("ai").write("Cleaning data...")
                msgs.add_ai_message("Cleaning data...")
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
                    
                    st.chat_message("ai").write("Data cleaning complete. Using cleaned data.")
                    msgs.add_ai_message("Data cleaning complete. Using cleaned data.")
                    current_df = cleaned_df
                else:
                    st.chat_message("ai").write("Data cleaning did not produce new data. Using original data.")
                    msgs.add_ai_message("Data cleaning did not produce new data. Using original data.")
            except Exception as dc_error:
                error_msg = f"An error occurred during data cleaning: {dc_error}. Using original data for analysis."
                st.chat_message("ai").write(error_msg)
                msgs.add_ai_message(error_msg)
        else:
            st.chat_message("ai").write("Skipping data cleaning as it doesn't appear necessary for your query.")
            msgs.add_ai_message("Skipping data cleaning as it doesn't appear necessary for your query.")

        # --- Feature Engineering Step (Conditional) ---
        if needs_feature_engineering:
            try:
                st.chat_message("ai").write("Performing feature engineering...")
                msgs.add_ai_message("Performing feature engineering...")
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
                    
                    st.chat_message("ai").write("Feature engineering complete. Using engineered data.")
                    msgs.add_ai_message("Feature engineering complete. Using engineered data.")
                    current_df = engineered_df
                else:
                    st.chat_message("ai").write("Feature engineering did not produce new data. Using current data.")
                    msgs.add_ai_message("Feature engineering did not produce new data. Using current data.")
            except Exception as fe_error:
                error_msg = f"An error occurred during feature engineering: {fe_error}. Using current data for analysis."
                st.chat_message("ai").write(error_msg)
                msgs.add_ai_message(error_msg)
        else:
            st.chat_message("ai").write("Skipping feature engineering as it doesn't appear necessary for your query.")
            msgs.add_ai_message("Skipping feature engineering as it doesn't appear necessary for your query.")

        try:
            pandas_data_analyst.invoke_agent(
                user_instructions=question,
                data_raw=current_df,
            )
            result = pandas_data_analyst.get_response()
        except Exception as e:
            st.chat_message("ai").write(
                "An error occurred while processing your query. Please try again."
            )
            msgs.add_ai_message(
                "An error occurred while processing your query. Please try again."
            )
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
            else:
                st.chat_message("ai").write("The agent did not return a valid chart.")
                msgs.add_ai_message("The agent did not return a valid chart.")

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
            else:
                st.chat_message("ai").write("No table data was returned by the agent.")
                msgs.add_ai_message("No table data was returned by the agent.")
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
            else:
                response_text = (
                    "An error occurred while processing your query. Please try again."
                )
                msgs.add_ai_message(response_text)
                st.chat_message("ai").write(response_text)
