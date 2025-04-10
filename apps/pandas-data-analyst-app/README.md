# Sigla - Pandas Data Analyst App

Aplikacja analityczna pozwalająca na analizę danych za pomocą sztucznej inteligencji.

## Funkcjonalności

Aplikacja oferuje trzy główne tryby działania:

1. **Analiza danych z czatem** - Standardowy tryb, w którym możesz zadawać pytania dotyczące danych w języku naturalnym i otrzymywać odpowiedzi, wizualizacje oraz analizy.

2. **Analiza z pliku konfiguracyjnego** - Tryb umożliwiający automatyczną analizę danych na podstawie pliku konfiguracyjnego, który zawiera strukturę i instrukcje dla generowanych raportów.

3. **Czatowanie z danymi** - Tryb umożliwiający bezpośrednią konwersację z danymi, zadawanie pytań i otrzymywanie kontekstowych odpowiedzi opartych na zawartości danych.

## Jak korzystać z aplikacji

### Uruchomienie aplikacji

1. Zainstaluj wymagane pakiety: `pip install -r requirements.txt`
2. Ustaw klucz API dla OpenAI w zmiennej środowiskowej `OPENAI_API_KEY` lub w pliku `.env`
3. Uruchom aplikację: `streamlit run app.py`

### Dane testowe

Aplikacja oferuje wbudowane dane testowe o sprzedaży rowerów, które można wykorzystać do przetestowania funkcjonalności bez konieczności wgrywania własnych plików. Aby skorzystać z danych testowych, wystarczy zaznaczyć opcję "Użyj przykładowych danych o sprzedaży rowerów" na ekranie głównym.

### Używanie trybu analizy z pliku konfiguracyjnego

W tym trybie możesz podać konfigurację analizy w formie tekstowej lub pliku, na podstawie której aplikacja automatycznie wygeneruje kompletny raport analityczny. Aplikacja oferuje również możliwość pobrania lub bezpośredniego użycia przykładowej konfiguracji.

**Format pliku konfiguracyjnego:**

```
Ogólny opis danych:
- Opis zbioru danych
- Dostępne kolumny: { df.columns.tolist() } # kolumny zostaną automatycznie dodane

Analiza danych:
Page 1:
    - Title: "Tytuł analizy"
    - Charts: "Opis wykresu do wygenerowania"
    - Description: "Opis analizy do przeprowadzenia"
Page 2:
    - Title: "Tytuł drugiej strony analizy"
    - Charts: "Opis drugiego wykresu"
    - Description: "Opis drugiej analizy"
...
```

### Używanie trybu czatowania z danymi

Tryb ten pozwala na interaktywną konwersację z danymi. Możesz zadawać pytania, a aplikacja wykorzysta kontekst danych, uprzednio przeprowadzone analizy oraz historię rozmowy, aby dostarczyć trafne odpowiedzi.

System automatycznie sugeruje również potencjalne pytania uzupełniające, które mogą pomóc w zgłębieniu analizowanego tematu.

## Przykłady pytań

- "Pokaż średnią 7-dniową kroczącą sprzedaży dla każdego modelu roweru"
- "Jaka jest korelacja między ceną a ilością sprzedanych produktów?"
- "Stwórz wykres przedstawiający rozkład sprzedaży w zależności od koloru produktu"
- "Jaki jest najlepiej sprzedający się produkt w kategorii premium?"
