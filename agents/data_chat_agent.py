# SIGLA
# AI DATA SCIENCE TEAM
# ***
# * Agents: Data Chat Agent

# Libraries
from typing import TypedDict, Annotated, Sequence, Literal, Union, Optional, Dict, List
import operator
import os
import json
import pandas as pd
from IPython.display import Markdown

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.types import Command, Checkpointer
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph

from ai_data_science_team.templates import (
    BaseAgent,
)
from ai_data_science_team.utils.regex import (
    get_generic_summary,
)
from ai_data_science_team.tools.dataframe import get_dataframe_summary
from ai_data_science_team.utils.logging import log_ai_function

# Setup
AGENT_NAME = "data_chat_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

# Data Chat Agent Graph State
class DataChatGraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data_raw: Union[dict, list]
    data_cleaned: Union[dict, list]
    chat_history: List[Dict]
    data_analysis_results: Dict
    data_description: str
    user_query: str
    system_response: str
    error_messages: Dict
    max_retries: int
    retry_count: int

class DataChatAgent(BaseAgent):
    """
    DataChatAgent enables natural language interactions with data.
    
    The agent allows users to ask questions about data and receive 
    context-aware responses based on the dataset content and any
    previously performed analysis.
    
    Parameters:
    -----------
    model:
        The language model to be used for the agent.
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
    invoke_agent(data_raw, data_cleaned, analysis_results, user_query, **kwargs)
        Invoke the agent with data, analysis results, and a user query.
    get_response()
        Get the agent's response to the last query.
    """
    
    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        checkpointer: Checkpointer = None
    ):
        self._params = {
            "model": model,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "checkpointer": checkpointer
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None
        self.chat_history = []
    
    def _make_compiled_graph(self):
        """Create or rebuild the compiled graph. Resets response to None."""
        self.response = None
        return make_data_chat_agent(**self._params)
    
    def update_params(self, **kwargs):
        """Updates parameters and rebuilds the compiled graph."""
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()
    
    def invoke_agent(
        self, 
        data_raw: pd.DataFrame, 
        data_cleaned: Optional[pd.DataFrame] = None,
        analysis_results: Optional[Dict] = None,
        user_query: str = "",
        max_retries: int = 3, 
        retry_count: int = 0, 
        **kwargs
    ):
        """
        Invokes the agent with data and a user query.
        
        Parameters:
        -----------
        data_raw: pd.DataFrame
            The raw dataset
        data_cleaned: pd.DataFrame, optional
            The cleaned dataset (if available)
        analysis_results: Dict, optional
            Results from previous analysis (if available)
        user_query: str
            The user's query about the data
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
        if data_cleaned is None:
            data_cleaned = data_raw
        
        if analysis_results is None:
            analysis_results = {}
        
        # Invoke the graph
        response = self._compiled_graph.invoke({
            "data_raw": self._convert_data_input(data_raw),
            "data_cleaned": self._convert_data_input(data_cleaned),
            "chat_history": self.chat_history,
            "data_analysis_results": analysis_results,
            "data_description": "",
            "user_query": user_query,
            "system_response": "",
            "error_messages": {},
            "max_retries": max_retries,
            "retry_count": retry_count
        }, **kwargs)
        
        # Update chat history
        if response.get("chat_history"):
            self.chat_history = response.get("chat_history")
        
        self.response = response
        return None
    
    def get_response(self):
        """Get the agent's response to the last query."""
        if self.response:
            return self.response.get("system_response", "")
    
    def get_chat_history(self):
        """Get the full chat history."""
        return self.chat_history
    
    def get_workflow_summary(self, markdown=False):
        """Retrieves the agent's workflow summary."""
        if self.response and self.response.get("messages"):
            summary = get_generic_summary(json.loads(self.response.get("messages")[-1].content))
            if markdown:
                return Markdown(summary)
            else:
                return summary
    
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

def make_data_chat_agent(
    model,
    n_samples=30,
    log=False,
    log_path=None,
    checkpointer=None
):
    """
    Creates a DataChatAgent that enables natural language interactions with data.
    
    Parameters:
    -----------
    model:
        The language model to use
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
    A compiled graph for the DataChatAgent
    """
    llm = model
    
    if log_path is None:
        log_path = LOG_PATH
    
    if log and not os.path.exists(log_path):
        os.makedirs(log_path)
    
    # Node for preparing data context
    def prepare_data_context(state: DataChatGraphState):
        """Prepare the data context for chat capabilities."""
        data_raw = state["data_raw"]
        data_cleaned = state["data_cleaned"] or data_raw
        
        # Generate data description
        data_df = pd.DataFrame.from_dict(data_cleaned)
        data_summary = get_dataframe_summary(data_df)
        
        # Create a comprehensive data description
        data_description_prompt = PromptTemplate.from_template("""
        Jesteś analitykiem danych badającym zbiór danych. 
        Na podstawie poniższego podsumowania zbioru danych, przedstaw kompleksowy opis 
        danych, który mógłby pomóc w odpowiadaniu na pytania na ich temat.
        
        Podsumowanie zbioru danych:
        {data_summary}
        
        Twój opis powinien zawierać:
        1. Co wydaje się zawierać ten zbiór danych
        2. Kluczowe zmienne i ich znaczenie
        3. Wszelkie zauważalne wzorce, rozkłady lub relacje
        4. Potencjalne spostrzeżenia, które można wyciągnąć z tych danych
        
        Podaj zwięzły, ale kompleksowy opis. Napisz go po polsku.
        """)
        
        data_description_response = llm.invoke(
            data_description_prompt.format(
                data_summary=data_summary
            )
        )
        
        # Extract insight from analysis results if available
        analysis_insights = ""
        if state["data_analysis_results"]:
            insights_prompt = PromptTemplate.from_template("""
            Jesteś analitykiem danych podsumowującym wyniki poprzednich analiz.
            Na podstawie poniższych wyników analiz, przedstaw zwięzłe podsumowanie
            kluczowych wniosków, które mogą być przydatne przy odpowiadaniu na przyszłe pytania dotyczące danych.
            
            Wyniki analiz:
            {analysis_results}
            
            Skup się na wyodrębnieniu najważniejszych ustaleń, które byłyby istotne
            dla przyszłych pytań o te dane. Zorganizuj wnioski według tematu lub kategorii.
            Napisz po polsku.
            """)
            
            insights_response = llm.invoke(
                insights_prompt.format(
                    analysis_results=json.dumps(state["data_analysis_results"], indent=2)
                )
            )
            
            analysis_insights = insights_response.content
        
        # Combine data description with analysis insights
        if analysis_insights:
            data_description = f"{data_description_response.content}\n\nPrevious Analysis Insights:\n{analysis_insights}"
        else:
            data_description = data_description_response.content
        
        return {
            "data_description": data_description
        }
    
    # Node for processing user query and generating response
    def process_query(state: DataChatGraphState):
        """Process the user query and generate a response based on the data context."""
        data_raw = state["data_raw"]
        data_cleaned = state["data_cleaned"] or data_raw
        data_description = state["data_description"]
        user_query = state["user_query"]
        chat_history = state["chat_history"]
        
        # Create chat history string
        chat_history_str = ""
        if chat_history:
            chat_history_str = "\n".join([
                f"User: {exchange.get('user', '')}\nAssistant: {exchange.get('assistant', '')}"
                for exchange in chat_history[-5:]  # Include only the last 5 exchanges
            ])
        
        # Process the query
        chat_prompt = ChatPromptTemplate.from_template("""
        Jesteś asystentem analizy danych, który pomaga użytkownikom zrozumieć i eksplorować dane.
        
        Opis danych:
        {data_description}
        
        Historia czatu:
        {chat_history}
        
        Na podstawie opisu danych i wszelkich dostarczonych wyników analiz, 
        odpowiedz na poniższe zapytanie w wyczerpujący sposób:

        Zapytanie użytkownika: {user_query}
        
        Wytyczne:
        1. Skup się na konkretnym zadanym pytaniu
        2. Odwołuj się do konkretnych punktów danych, gdy to możliwe
        3. Jeśli zapytanie wymaga obliczeń lub analiz, które nie zostały jeszcze wykonane, 
           wyjaśnij, co byłoby potrzebne
        4. Jeśli odpowiedź nie jest bezpośrednio dostępna z opisu danych lub 
           poprzedniej analizy, przyznaj to i zasugeruj, jak można by uzyskać te informacje
        
        Twoja odpowiedź powinna być jasna, oparta na faktach i bezpośrednio odnosić się do zapytania użytkownika.
        Odpowiedz po polsku.
        """)
        
        # Get data summary for reference
        data_df = pd.DataFrame.from_dict(data_cleaned)
        data_summary = get_dataframe_summary(data_df)
        
        # Invoke the LLM to generate a response
        chat_response = llm.invoke(
            chat_prompt.format(
                data_description=data_description,
                chat_history=chat_history_str,
                user_query=user_query
            )
        )
        
        # Update chat history
        updated_chat_history = chat_history + [{
            "user": user_query,
            "assistant": chat_response.content
        }]
        
        # Generate potential follow-up questions
        followup_prompt = PromptTemplate.from_template("""
        Na podstawie poniższego zapytania użytkownika i Twojej odpowiedzi,
        zaproponuj 2-3 potencjalne pytania uzupełniające, które użytkownik mógłby chcieć zadać.
        
        Zapytanie użytkownika: {user_query}
        Twoja odpowiedź: {response}
        Opis danych: {data_summary}
        
        Sformatuj swoje sugestie jako listę oddzieloną przecinkami.
        Napisz pytania po polsku.
        """)
        
        followup_response = llm.invoke(
            followup_prompt.format(
                user_query=user_query,
                response=chat_response.content,
                data_summary=data_summary
            )
        )
        
        # Process the response
        system_response = {
            "answer": chat_response.content,
            "suggested_followups": followup_response.content.split(",")
        }
        
        return {
            "system_response": system_response,
            "chat_history": updated_chat_history,
            "messages": [BaseMessage(
                content=json.dumps({
                    "agent": AGENT_NAME,
                    "status": "completed",
                    "query": user_query,
                    "response_length": len(chat_response.content)
                }),
                type="ai",
                role="assistant"
            )]
        }
    
    # Create the data chat agent graph
    workflow = StateGraph(DataChatGraphState)
    
    # Add nodes
    workflow.add_node("prepare_context", prepare_data_context)
    workflow.add_node("process_query", process_query)
    
    # Add edges
    workflow.add_edge(START, "prepare_context")
    workflow.add_edge("prepare_context", "process_query")
    workflow.add_edge("process_query", END)
    
    # Compile the graph
    return workflow.compile(checkpointer=checkpointer) 