# Sigla-Agent

System analityczny wykorzystujący LLM do analizy i wizualizacji danych z wykorzystaniem Pandas i Plotly.

## Architektura aplikacji

Sigla-Agent wykorzystuje architekturę opartą na LangGraph, dzięki czemu można efektywnie orkiestrować przepływ danych pomiędzy agentami specjalizującymi się w różnych zadaniach przetwarzania danych.

## Integracja z LangSmith

Aplikacja zawiera integrację z LangSmith dla zwiększonej obserwowalności i debugowania. Pomaga to zrozumieć, co dzieje się "pod maską", gdy AI przetwarza dane.

### Funkcje

- **Szczegółowe śledzenie**: Każde wywołanie LLM i operacja na danych jest rejestrowana z pełnym kontekstem
- **Śledzenie zużycia tokenów**: Monitorowanie kosztów i wydajności
- **Historia uruchomień**: Śledzenie wszystkich uruchomień wraz z ich wejściami i wyjściami
- **Zbieranie opinii**: Ocenianie wyników w celu ulepszenia aplikacji
- **Debugowanie**: Dokładny wgląd w generowany kod i sposób transformacji danych

### Konfiguracja

1. Uzyskaj klucz API LangSmith z [smith.langchain.com](https://smith.langchain.com)
2. Dodaj go do pliku `.env`:
   ```
   LANGSMITH_API_KEY=twój_klucz_tutaj
   ```
3. Włącz śledzenie w pasku bocznym podczas uruchamiania aplikacji

### Przeglądanie śladów

- Każde uruchomienie otrzymuje unikalny identyfikator Run ID wyświetlany w pasku bocznym
- Kliknij link, aby zobaczyć szczegółowe informacje o śladzie
- Używaj dashboardu LangSmith do analizy uruchomień, porównywania wydajności i identyfikacji błędów

## Tryby aplikacji

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Aplikacja Streamlit                           │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Tryby aplikacji                             │
├───────────────────────┬───────────────────────┬─────────────────────┤
│ Analiza i czatowanie  │ Analiza z pliku       │                     │
│ z danymi              │ konfiguracyjnego      │                     │
└─────────┬─────────────┴───────────┬───────────┴─────────────────────┘
          │                         │
          ▼                         ▼
┌────────────────────┐   ┌────────────────────┐
│ PandasDataAnalyst  │   │ Configuration      │
│ & DataChatAgent    │   │ AnalysisAgent      │
└────────┬───────────┘   └────────┬───────────┘
         │                        │
         │                        │
┌────────▼───────────────────────▼───────────────┐
│               Przepływ agentów                 │
├────────────────┬────────────────┬──────────────┤
│ DataCleaning   │ Feature        │ Data         │
│ Agent          │ Engineering    │ Visualization│
│                │ Agent          │ Agent        │
└────────────────┴────────────────┴──────────────┘
```

## Opis przepływu pracy

1. **Aplikacja Streamlit** - Główny interfejs aplikacji, który udostępnia różne tryby analizy danych.

2. **Tryby aplikacji**:
   - **Analiza i czatowanie z danymi** - Tryb umożliwiający zarówno analizę danych jak i konwersacyjny interfejs z PandasDataAnalyst i DataChatAgent.
   - **Analiza z pliku konfiguracyjnego** - Generuje kompletne raporty na podstawie pliku konfiguracyjnego.

3. **Główne agenty**:
   - **PandasDataAnalyst** - Orkiestruje przetwarzanie danych i wizualizację dla pytań ad-hoc.
   - **ConfigurationAnalysisAgent** - Przetwarza dane zgodnie ze strukturalnymi plikami konfiguracyjnymi zawierającymi predefiniowane analizy.
   - **DataChatAgent** - Zapewnia konwersacyjny interfejs do danych z pamięcią poprzednich pytań.

4. **Przepływ agentów** - Współdzielone komponenty używane przez różne agenty:
   - **DataCleaningAgent** - Obsługuje czyszczenie i wstępne przetwarzanie danych.
   - **FeatureEngineeringAgent** - Tworzy nowe cechy i transformuje dane.
   - **DataVisualizationAgent** - Generuje wykresy i wizualizacje.

Każdy agent wykorzystuje LangGraph do orkiestracji swojego przepływu pracy, gdzie węzły reprezentują określone zadania, a krawędzie kontrolują przepływ między nimi.

## Instalacja i uruchomienie

### Wymagania systemowe
- Python 3.12 lub nowszy
- [UV Package Manager](https://github.com/astral-sh/uv)

### Konfiguracja

1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/twojadres/Sigla-Agent.git
   cd Sigla-Agent
   ```

2. Utwórz środowisko wirtualne i zainstaluj zależności:
   ```bash
   uv venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

3. Skonfiguruj plik `.env` z kluczami API:
   ```
   OPENAI_API_KEY=twój_klucz_openai
   LANGSMITH_API_KEY=twój_klucz_langsmith  # opcjonalnie
   ENABLE_LANGSMITH=true  # opcjonalnie, domyślnie false
   ```

### Uruchomienie aplikacji

```bash
cd apps/pandas-data-analyst-app
streamlit run app.py
```

## Zależności

Projekt wykorzystuje następujące główne biblioteki:
- `langchain` i `langgraph` - orkiestracja agentów AI
- `openai` - dostęp do modeli LLM
- `pandas` i `numpy` - przetwarzanie danych
- `plotly` - wizualizacja danych
- `streamlit` - interfejs użytkownika
- `ai-data-science-team` - biblioteka z podstawowymi agentami analitycznymi

Pełna lista zależności znajduje się w pliku `pyproject.toml`.
