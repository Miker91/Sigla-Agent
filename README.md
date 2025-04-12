# Agent Architecture Overview

# Sigla-Agent

## LangSmith Observability

The application now includes integration with LangSmith for enhanced observability and debugging. This helps you understand what's happening under the hood when the AI processes your data.

### Features

- **Detailed Tracing**: Every LLM call and data operation is logged with full context
- **Token Usage Tracking**: Monitor cost and performance
- **Run History**: Track all runs with their inputs and outputs
- **Feedback Collection**: Rate results to improve the application
- **Debugging**: See exactly what code was generated and how data was transformed

### Setup

1. Get a LangSmith API key from [smith.langchain.com](https://smith.langchain.com)
2. Add it to your `.env` file:
   ```
   LANGSMITH_API_KEY=your_key_here
   ```
3. Enable tracing in the sidebar when running the application

### Viewing Traces

- Each run gets a unique Run ID displayed in the sidebar
- Click on the link to view detailed trace information
- Use the LangSmith dashboard to analyze runs, compare performance, and identify errors

## Main Application Modes

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit Application                        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Application Modes                           │
├───────────────────────┬───────────────────────┬─────────────────────┤
│ Analiza danych z      │ Analiza z pliku       │ Czatowanie z        │
│ czatem                │ konfiguracyjnego      │ danymi              │
└─────────┬─────────────┴───────────┬───────────┴──────────┬──────────┘
          │                         │                      │
          ▼                         ▼                      ▼
┌────────────────────┐   ┌────────────────────┐   ┌───────────────────┐
│ PandasDataAnalyst  │   │ Configuration      │   │ DataChatAgent     │
│                    │   │ AnalysisAgent      │   │                   │
└────────┬───────────┘   └────────┬───────────┘   └─────────┬─────────┘
         │                        │                         │
         │                        │                         │
┌────────▼───────────────────────▼───────────────┐   ┌─────▼─────────┐
│               Agent Workflow                   │   │ Chat Workflow │
├────────────────┬────────────────┬──────────────┤   ├───────────────┤
│ DataCleaning   │ Feature        │ Data         │   │ prepare_context│
│ Agent          │ Engineering    │ Visualization│   ├───────────────┤
│                │ Agent          │ Agent        │   │ process_query  │
└────────────────┴────────────────┴──────────────┘   └───────────────┘
```

## Workflow Description

1. **Streamlit Application** - The main application interface that provides three different modes for data analysis.

2. **Application Modes**:
   - **Analiza danych z czatem** (Data Analysis with Chat) - Standard mode where users ask questions and get answers through PandasDataAnalyst.
   - **Analiza z pliku konfiguracyjnego** (Configuration-based Analysis) - Generates complete reports based on a configuration file.
   - **Czatowanie z danymi** (Chat with Data) - Direct conversational interface with the data.

3. **Core Agents**:
   - **PandasDataAnalyst** - Orchestrates data wrangling and visualization for ad-hoc questions.
   - **ConfigurationAnalysisAgent** - Processes data according to structured configuration files with predefined analyses.
   - **DataChatAgent** - Provides conversational interface to data with memory of previous questions.

4. **Agent Workflow** - The shared components used by different agents:
   - **DataCleaningAgent** - Handles data cleaning and preprocessing.
   - **FeatureEngineeringAgent** - Creates new features and transforms data.
   - **DataVisualizationAgent** - Generates charts and visualizations.

5. **Chat Workflow** - Specific to the DataChatAgent:
   - **prepare_context** - Analyzes data to build context for answering questions.
   - **process_query** - Generates responses based on the data and question context.

Each agent uses LangGraph for orchestrating its workflow, with nodes representing specific tasks and edges controlling the flow between them.
