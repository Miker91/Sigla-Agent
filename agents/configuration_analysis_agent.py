# SIGLA
# AI DATA SCIENCE TEAM
# ***
# * Agents: Configuration Analysis Agent

# Libraries
from typing import TypedDict, Annotated, Sequence, Literal, Union, Optional, Dict, List
import operator
import os
import json
import yaml
import pandas as pd
from IPython.display import Markdown

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.types import Command, Checkpointer
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph

from ai_data_science_team.templates import (
    node_func_execute_agent_code_on_data, 
    node_func_human_review,
    node_func_fix_agent_code, 
    node_func_report_agent_outputs,
    create_coding_agent_graph,
    BaseAgent,
)
from ai_data_science_team.parsers.parsers import PythonOutputParser
from ai_data_science_team.utils.regex import (
    relocate_imports_inside_function, 
    add_comments_to_top, 
    format_agent_name, 
    get_generic_summary,
)
from ai_data_science_team.tools.dataframe import get_dataframe_summary
from ai_data_science_team.utils.logging import log_ai_function

# Setup
AGENT_NAME = "configuration_analysis_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

# Configuration Analysis Agent Graph State
class ConfigAnalysisGraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    config_file_path: str
    config_data: Dict
    data_raw: Union[dict, list]
    data_cleaned: Union[dict, list]
    data_analysis_results: Dict
    current_page: int
    total_pages: int
    analysis_code: Dict
    analysis_visualizations: Dict
    analysis_descriptions: Dict
    error_messages: Dict
    max_retries: int
    retry_count: int

class ConfigurationAnalysisAgent(BaseAgent):
    """
    ConfigurationAnalysisAgent is responsible for analyzing data based on a configuration file.
    
    The agent reads a configuration file that specifies:
    1. General description of the data
    2. Pages of analysis to be performed, each with:
       - Title
       - Charts to be created
       - Description/analysis to be generated
    
    Parameters:
    -----------
    model:
        The language model to be used for the agent.
    data_cleaning_agent: 
        Optional data cleaning agent to use for data preprocessing.
    data_visualization_agent:
        Data visualization agent for creating charts.
    n_samples: int
        Number of samples to use for data summary.
    log: bool
        Whether to log agent operations.
    log_path: str
        Path to store logs.
    checkpointer: Checkpointer (optional)
        Checkpointer for the agent.
    
    Methods:
    --------
    invoke_agent(data_raw, config_file_path, **kwargs)
        Invoke the agent with data and a configuration file.
    get_analysis_results()
        Get the complete analysis results.
    get_page_analysis(page_number)
        Get the analysis for a specific page.
    """
    
    def __init__(
        self,
        model,
        data_cleaning_agent=None,
        data_visualization_agent=None,
        n_samples=30,
        log=False,
        log_path=None,
        checkpointer: Checkpointer = None
    ):
        self._params = {
            "model": model,
            "data_cleaning_agent": data_cleaning_agent,
            "data_visualization_agent": data_visualization_agent,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "checkpointer": checkpointer
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None
    
    def _make_compiled_graph(self):
        """Create or rebuild the compiled graph. Resets response to None."""
        self.response = None
        return make_configuration_analysis_agent(**self._params)
    
    def update_params(self, **kwargs):
        """Updates parameters and rebuilds the compiled graph."""
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()
    
    def invoke_agent(self, data_raw: pd.DataFrame, config_file_path: str, max_retries: int = 3, retry_count: int = 0, **kwargs):
        """
        Invokes the agent with a dataset and configuration file.
        
        Parameters:
        -----------
        data_raw: pd.DataFrame
            The raw dataset to be analyzed
        config_file_path: str
            Path to the configuration file
        max_retries: int
            Maximum number of retry attempts
        retry_count: int
            Current retry count
        **kwargs
            Additional arguments to pass to the invoke method
        
        Returns:
        --------
        None. Results stored in the response attribute.
        """
        # Load and parse configuration file
        with open(config_file_path, 'r') as f:
            if config_file_path.endswith('.yaml') or config_file_path.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                # Try to parse as JSON or as plain text
                try:
                    config_data = json.load(f)
                except json.JSONDecodeError:
                    # If not valid JSON, treat as plain text
                    f.seek(0)  # Reset file pointer
                    config_data = {'raw_text': f.read()}
        
        # Preprocess configuration for columns information
        if isinstance(config_data, dict) and 'raw_text' in config_data:
            # For text-based configs, replace placeholder for columns
            raw_text = config_data['raw_text']
            if '{ df.columns.tolist() }' in raw_text:
                columns_str = str(data_raw.columns.tolist())
                raw_text = raw_text.replace('{ df.columns.tolist() }', columns_str)
                config_data['raw_text'] = raw_text
        
        # Invoke the graph
        response = self._compiled_graph.invoke({
            "config_file_path": config_file_path,
            "config_data": config_data,
            "data_raw": self._convert_data_input(data_raw),
            "data_cleaned": None,
            "data_analysis_results": {},
            "current_page": 0,
            "total_pages": 0,
            "analysis_code": {},
            "analysis_visualizations": {},
            "analysis_descriptions": {},
            "error_messages": {},
            "max_retries": max_retries,
            "retry_count": retry_count
        }, **kwargs)
        
        self.response = response
        return None
    
    def get_workflow_summary(self, markdown=False):
        """Retrieves the agent's workflow summary."""
        if self.response and self.response.get("messages"):
            summary = get_generic_summary(json.loads(self.response.get("messages")[-1].content))
            if markdown:
                return Markdown(summary)
            else:
                return summary
    
    def get_analysis_results(self):
        """Get the complete analysis results for all pages."""
        if self.response:
            return self.response.get("data_analysis_results")
    
    def get_page_analysis(self, page_number):
        """Get analysis results for a specific page."""
        if self.response and self.response.get("data_analysis_results"):
            results = self.response.get("data_analysis_results")
            page_key = f"Page {page_number}"
            if page_key in results:
                return results[page_key]
    
    def get_total_pages(self):
        """Get the total number of pages in the analysis."""
        if self.response:
            return self.response.get("total_pages", 0)
    
    def get_data_cleaned(self):
        """Get the cleaned dataset used for analysis."""
        if self.response and self.response.get("data_cleaned"):
            return pd.DataFrame(self.response.get("data_cleaned"))
    
    def get_data_raw(self):
        """Get the raw dataset."""
        if self.response and self.response.get("data_raw"):
            return pd.DataFrame(self.response.get("data_raw"))
    
    def get_visualizations(self):
        """Get all visualizations generated in the analysis."""
        if self.response:
            return self.response.get("analysis_visualizations", {})
    
    def get_description(self, page_number):
        """Get the descriptive analysis for a specific page."""
        if self.response and self.response.get("analysis_descriptions"):
            descriptions = self.response.get("analysis_descriptions")
            page_key = f"Page {page_number}"
            if page_key in descriptions:
                return descriptions[page_key]
    
    @staticmethod
    def _convert_data_input(data_raw: Union[pd.DataFrame, dict, list]) -> Union[dict, list]:
        """Converts input data to the expected format (dict or list of dicts)."""
        if isinstance(data_raw, pd.DataFrame):
            return data_raw.to_dict()
        if isinstance(data_raw, dict):
            return data_raw
        if isinstance(data_raw, list):
            return [item.to_dict() if isinstance(item, pd.DataFrame) else item for item in data_raw]
        raise ValueError("data_raw must be a DataFrame, dict, or list of DataFrames/dicts")

def make_configuration_analysis_agent(
    model,
    data_cleaning_agent=None,
    data_visualization_agent=None,
    n_samples=30,
    log=False,
    log_path=None,
    checkpointer=None
):
    """
    Creates a ConfigurationAnalysisAgent that processes data analysis based on a configuration file.
    
    Parameters:
    -----------
    model:
        The language model to use
    data_cleaning_agent:
        Optional data cleaning agent for preprocessing
    data_visualization_agent:
        Data visualization agent for creating charts
    n_samples: int
        Number of samples to use for data summary
    log: bool
        Whether to log operations
    log_path: str
        Path to store logs
    checkpointer: Checkpointer
        Checkpointer for the agent
    
    Returns:
    --------
    A compiled graph for the ConfigurationAnalysisAgent
    """
    llm = model
    
    if log_path is None:
        log_path = LOG_PATH
    
    if log and not os.path.exists(log_path):
        os.makedirs(log_path)
    
    # Node for parsing configuration and preparing analysis
    def parse_config_and_prepare(state: ConfigAnalysisGraphState):
        """Parse the configuration file and prepare the analysis plan."""
        config_data = state["config_data"]
        data_raw = state["data_raw"]
        
        # For plain text configs, use LLM to structure the data
        if isinstance(config_data, dict) and 'raw_text' in config_data:
            raw_text = config_data['raw_text']
            
            parse_config_prompt = PromptTemplate.from_template("""
            You are a data analyst preparing an analysis plan based on a configuration file.
            Please parse the following configuration text and convert it to a structured format.
            
            Configuration text:
            ```
            {config_text}
            ```
            
            Extract the following information:
            1. General description of the data
            2. For each page:
               - Page number
               - Title
               - Charts to be created
               - Description or analysis to be performed
            
            Return a JSON object with this structure:
            ```json
            {{
                "general_description": "Description of the data",
                "pages": [
                    {{
                        "page_number": 1,
                        "title": "Page Title",
                        "charts": "Chart description",
                        "description": "Analysis description"
                    }},
                    ...
                ]
            }}
            ```
            
            Only return the JSON object, nothing else.
            """)
            
            structured_config_response = llm.invoke(
                parse_config_prompt.format(config_text=raw_text)
            )
            
            # Extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', structured_config_response.content, re.DOTALL)
            if json_match:
                structured_config = json.loads(json_match.group(0))
            else:
                structured_config = {"general_description": "", "pages": []}
        else:
            # If already structured, use as is
            structured_config = config_data
        
        # Count total pages for analysis
        total_pages = len(structured_config.get("pages", []))
        
        # Initialize empty results for each page
        analysis_code = {}
        analysis_visualizations = {}
        analysis_descriptions = {}
        
        return {
            "config_data": structured_config,
            "total_pages": total_pages,
            "current_page": 1 if total_pages > 0 else 0,
            "analysis_code": analysis_code,
            "analysis_visualizations": analysis_visualizations,
            "analysis_descriptions": analysis_descriptions
        }
    
    # Node for cleaning data if a cleaning agent is provided
    def clean_data(state: ConfigAnalysisGraphState):
        """Clean the data using the provided data cleaning agent if available."""
        if data_cleaning_agent is None:
            # No cleaning agent, use raw data as cleaned data
            return {"data_cleaned": state["data_raw"]}
        
        # Use the cleaning agent to clean the data
        try:
            # Get structured config
            config_data = state["config_data"]
            
            # Generate cleaning instructions from config
            cleaning_prompt = PromptTemplate.from_template("""
            You are a data cleaning specialist preparing data for analysis.
            Based on the following data description and analysis requirements, 
            provide comprehensive data cleaning instructions.
            
            Data description: {general_description}
            
            Analysis requirements:
            {analysis_requirements}
            
            Provide detailed data cleaning instructions to prepare this dataset for these analyses.
            Focus on ensuring data quality, handling missing values, correcting data types,
            and any transformations needed for the specified charts and analyses.
            """)
            
            # Extract analysis requirements for all pages
            pages = config_data.get("pages", [])
            analysis_requirements = "\n".join([
                f"Page {page.get('page_number', i+1)}: {page.get('charts', '')}"
                for i, page in enumerate(pages)
            ])
            
            cleaning_instructions = llm.invoke(
                cleaning_prompt.format(
                    general_description=config_data.get("general_description", ""),
                    analysis_requirements=analysis_requirements
                )
            ).content
            
            # Invoke the data cleaning agent
            data_raw_df = pd.DataFrame.from_dict(state["data_raw"])
            data_cleaning_agent.invoke_agent(
                user_instructions=cleaning_instructions,
                data_raw=data_raw_df
            )
            
            # Get the cleaned data
            cleaned_df = data_cleaning_agent.get_data_cleaned()
            
            if cleaned_df is not None:
                return {"data_cleaned": cleaned_df.to_dict()}
            else:
                # If cleaning failed, use raw data
                return {"data_cleaned": state["data_raw"]}
            
        except Exception as e:
            # If any error occurs, use raw data
            error_msg = f"Error during data cleaning: {str(e)}"
            return {
                "data_cleaned": state["data_raw"],
                "error_messages": {**state["error_messages"], "cleaning": error_msg}
            }
    
    # Node for analyzing the current page
    def analyze_page(state: ConfigAnalysisGraphState):
        """Analyze the current page according to the configuration."""
        current_page = state["current_page"]
        config_data = state["config_data"]
        data_cleaned = state["data_cleaned"]
        
        # If all pages are processed, return results
        if current_page > state["total_pages"]:
            return {"messages": [BaseMessage(content=json.dumps({
                "agent": AGENT_NAME,
                "status": "completed",
                "total_pages_analyzed": state["total_pages"]
            }), role="assistant")]}
        
        # Get the page configuration
        pages = config_data.get("pages", [])
        page_config = next(
            (page for page in pages if page.get("page_number", 0) == current_page),
            pages[current_page - 1] if 0 < current_page <= len(pages) else None
        )
        
        if not page_config:
            return {
                "current_page": current_page + 1,
                "error_messages": {
                    **state["error_messages"], 
                    f"Page {current_page}": "Page configuration not found"
                }
            }
        
        # Get page information
        title = page_config.get("title", f"Analysis Page {current_page}")
        charts_desc = page_config.get("charts", "")
        analysis_desc = page_config.get("description", "")
        
        # Generate visualization code for the current page
        if data_visualization_agent:
            try:
                # Prepare the visualization instruction
                viz_instruction = f"""
                Create a visualization based on the following requirements:
                
                Chart description: {charts_desc}
                
                Use the dataset provided to create this visualization.
                """
                
                # Invoke the data visualization agent
                data_df = pd.DataFrame.from_dict(data_cleaned)
                data_visualization_agent.invoke_agent(
                    user_instructions=viz_instruction,
                    data_raw=data_df
                )
                
                # Get the visualization result
                plotly_graph = data_visualization_agent.get_plotly_graph()
                visualization_code = data_visualization_agent.get_data_visualization_function()
                
                # Store the visualization
                analysis_visualizations = {
                    **state["analysis_visualizations"],
                    f"Page {current_page}": plotly_graph
                }
                analysis_code = {
                    **state["analysis_code"],
                    f"Page {current_page}": visualization_code
                }
            except Exception as e:
                error_msg = f"Error generating visualization for Page {current_page}: {str(e)}"
                analysis_visualizations = state["analysis_visualizations"]
                analysis_code = state["analysis_code"]
                state["error_messages"] = {**state["error_messages"], f"Page {current_page}_viz": error_msg}
        else:
            # No visualization agent, generate code directly
            viz_prompt = PromptTemplate.from_template("""
            You are a data visualization expert.
            
            Create Python code to generate a visualization based on these requirements:
            
            Chart description: {charts_desc}
            
            Use the Plotly library to create the visualization.
            The data is available as a pandas DataFrame called 'df' with the following structure:
            
            {data_summary}
            
            Return ONLY the Python code to create this visualization. The code should:
            1. Be a single function that takes a DataFrame as input and returns a Plotly figure
            2. Include all necessary imports inside the function
            3. Only use the Plotly library for visualization
            
            The function should have this signature:
            ```python
            def create_visualization(df):
                # Your code here
                return fig
            ```
            """)
            
            # Get data summary
            data_df = pd.DataFrame.from_dict(data_cleaned)
            data_summary = get_dataframe_summary(data_df)
            
            # Generate visualization code
            viz_code_response = llm.invoke(
                viz_prompt.format(
                    charts_desc=charts_desc,
                    data_summary=data_summary
                )
            )
            
            # Extract code from response
            import re
            code_match = re.search(r'```python\s*(.*?)```', viz_code_response.content, re.DOTALL)
            if code_match:
                viz_code = code_match.group(1)
            else:
                viz_code = viz_code_response.content
            
            # Store the visualization code
            analysis_code = {
                **state["analysis_code"],
                f"Page {current_page}": viz_code
            }
            analysis_visualizations = state["analysis_visualizations"]
        
        # Generate analysis description
        analysis_prompt = PromptTemplate.from_template("""
        You are a data analyst interpreting visualization results.
        
        Based on the visualization of this data and the requirements below,
        provide a detailed analysis and interpretation:
        
        Chart type: {charts_desc}
        Analysis requirements: {analysis_desc}
        
        Data summary:
        {data_summary}
        
        Provide a comprehensive analysis addressing the requirements above.
        Your response should include:
        1. Interpretation of the main trends or patterns visible in the chart
        2. Key insights that can be derived from this visualization
        3. Specific conclusions relevant to the analysis requirements
        4. Any potential recommendations based on these findings
        
        Keep your analysis factual and data-driven.
        """)
        
        # Get data summary if not already obtained
        if 'data_summary' not in locals():
            data_df = pd.DataFrame.from_dict(data_cleaned) 
            data_summary = get_dataframe_summary(data_df)
        
        # Generate analysis description
        analysis_response = llm.invoke(
            analysis_prompt.format(
                charts_desc=charts_desc,
                analysis_desc=analysis_desc,
                data_summary=data_summary
            )
        )
        
        # Store analysis description
        analysis_descriptions = {
            **state["analysis_descriptions"],
            f"Page {current_page}": analysis_response.content
        }
        
        # Store results for the current page
        data_analysis_results = {
            **state.get("data_analysis_results", {}),
            f"Page {current_page}": {
                "title": title,
                "charts_description": charts_desc,
                "analysis_description": analysis_response.content,
                "visualization_code": analysis_code.get(f"Page {current_page}", ""),
                "has_visualization": f"Page {current_page}" in analysis_visualizations
            }
        }
        
        # Move to the next page
        return {
            "current_page": current_page + 1,
            "analysis_code": analysis_code,
            "analysis_visualizations": analysis_visualizations,
            "analysis_descriptions": analysis_descriptions,
            "data_analysis_results": data_analysis_results
        }
    
    # Create the configuration analysis agent graph
    workflow = StateGraph(ConfigAnalysisGraphState)
    
    # Add nodes
    workflow.add_node("parse_config", parse_config_and_prepare)
    workflow.add_node("clean_data", clean_data)
    workflow.add_node("analyze_page", analyze_page)
    
    # Add edges
    workflow.add_edge(START, "parse_config")
    workflow.add_edge("parse_config", "clean_data")
    workflow.add_edge("clean_data", "analyze_page")
    
    # Add conditional edges using routes
    workflow.add_conditional_edges(
        "analyze_page",
        lambda state: "analyze_page" if state["current_page"] <= state["total_pages"] else END
    )
    
    # Compile the graph
    return workflow.compile(checkpointer=checkpointer) 