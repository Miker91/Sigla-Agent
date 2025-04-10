
# Agent Architecture Overview

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
