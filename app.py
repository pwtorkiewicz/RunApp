import streamlit as st
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from typing import Literal, Annotated, Optional
import json
import re
import pandas as pd
from pycaret.regression import load_model, predict_model # type: ignore
from dotenv import dotenv_values
from langfuse import Langfuse
from langfuse.decorators import observe
from langfuse.openai import OpenAI as LangfuseOpenAI
import boto3
import io
import pickle
import sys # Importujemy sys do wypisywania na stderr, co często jest logowane
import os


# --- Inicjalizacja zmiennych środowiskowych, ścieżki modelu ML, bucketu w DigitalOcean ---

#env = dotenv_values(".env")
env_file = dotenv_values(".env")
env = {**env_file, **os.environ}
Model_pkl_in_spaces = "RunModel/Model/runtime_regression_pipeline.pkl" # Zastosowany model ML na Digital Ocean Spaces
#Model_pkl_in_github = "Model/runtime_regression_pipeline" # Zastosowany model ML na Github
BUCKET_NAME = "civil-eng"
#test
# --- ustawienie set_page_config Streamlit ---

st.set_page_config(page_title="Przewidywanie Czasu Półmaratonu")

# --- Inicjalizacja stanu sesji dla Langfuse i OpenAI ---
# Inicjalizacja wszystkich potrzebnych zmiennych w st.session_state na początku.
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None
if "user_input_text_area" not in st.session_state:
    st.session_state["user_input_text_area"] = ""
if "show_results" not in st.session_state:
    st.session_state["show_results"] = False


# OpenAI API key protection
if not st.session_state["openai_api_key"]:
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]
    else:
        st.info("Dodaj swój klucz API OpenAI, aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password", key="openai_key_input")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state["openai_api_key"]:
    st.stop()

# Inicjalizacja klienta OpenAI z biblioteką instructor
client = OpenAI(api_key=st.session_state["openai_api_key"])
instructor_openai_client = instructor.from_openai(client)

# Inicjacja Langfuse
langfuse = Langfuse(
    public_key=env.get("LANGFUSE_PUBLIC_KEY"),
    secret_key=env.get("LANGFUSE_SECRET_KEY"),
    host=env.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
)
langfuse.auth_check()

# --- Ładowanie Modelu ML---

@st.cache_resource
def load_pycaret_model_from_do_spaces(bucket_name: str, object_key: str):
    """
    Ładuje model PyCaret (.pkl) z DigitalOcean Spaces, 
    używając kluczy i endpointu ze zmiennych środowiskowych.

    Args:
        bucket_name (str): Nazwa bucketa w DigitalOcean Spaces, gdzie przechowywany jest model.
        object_key (str): Klucz obiektu (ścieżka do pliku) modelu w bucketcie.
                            Np. "RunModel/Model/runtime_regression_pipeline.pkl"

    Returns:
        object: Załadowany model PyCaret.
        None: Jeśli wystąpi błąd podczas ładowania.
    """
    st.info("Próba załadowania modelu z DigitalOcean Spaces...")
    print("DEBUG: Rozpoczynam ładowanie modelu z DO Spaces.", file=sys.stderr)

    # Pobieranie danych uwierzytelniających i endpointu ze zmiennych środowiskowych
    endpoint_url = env.get("AWS_ENDPOINT_URL_S3")
    aws_access_key_id = env.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = env.get("AWS_SECRET_ACCESS_KEY")

    print(f"DEBUG: Endpoint URL: {endpoint_url}", file=sys.stderr)
    print(f"DEBUG: AWS Access Key ID (pierwsze 4 znaki): {aws_access_key_id[:4] if aws_access_key_id else 'Brak'}", file=sys.stderr)
    # Nie logujemy całego secret_access_key ze względów bezpieczeństwa

    # Sprawdzenie, czy wszystkie zmienne są dostępne
    if not all([endpoint_url, aws_access_key_id, aws_secret_access_key]):
        missing_vars = []
        if not endpoint_url: missing_vars.append("AWS_ENDPOINT_URL_S3")
        if not aws_access_key_id: missing_vars.append("AWS_ACCESS_KEY_ID")
        if not aws_secret_access_key: missing_vars.append("AWS_SECRET_ACCESS_KEY")
        
        error_msg = f"Błąd: Brakujące zmienne środowiskowe: {', '.join(missing_vars)}. Sprawdź plik .env lub konfigurację środowiska na DigitalOcean."
        st.error(error_msg)
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return None

    try:
        print("DEBUG: Inicjalizuję klienta S3...", file=sys.stderr)
        # Inicjalizacja klienta S3 dla DigitalOcean Spaces z danych ze zmiennych środowiskowych
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        print("DEBUG: Klient S3 zainicjalizowany pomyślnie.", file=sys.stderr)

        print(f"DEBUG: Próbuję pobrać obiekt '{object_key}' z bucketa '{bucket_name}'...", file=sys.stderr)
        # Pobieranie obiektu z Spaces
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        print(f"DEBUG: Obiekt '{object_key}' pobrany pomyślnie.", file=sys.stderr)
        
        # Odczytywanie zawartości pliku jako strumienia bajtów
        print("DEBUG: Odczytuję zawartość pliku do strumienia bajtów...", file=sys.stderr)
        model_bytes = io.BytesIO(response['Body'].read())
        print("DEBUG: Zawartość pliku odczytana. Rozmiar (pierwsze 100 bajtów):", len(model_bytes.getvalue()[:100]), file=sys.stderr) # Sprawdzenie, czy coś zostało odczytane
        
        # Deserializacja modelu PyCaret
        print("DEBUG: Rozpoczynam deserializację modelu PyCaret...", file=sys.stderr)
        model = pickle.load(model_bytes)
        print("DEBUG: Model PyCaret zdeserializowany pomyślnie.", file=sys.stderr)
        
        st.success(f"Model '{object_key}' załadowany pomyślnie z bucketa '{bucket_name}'.")
        print(f"INFO: Model '{object_key}' załadowany pomyślnie z bucketa '{bucket_name}'.", file=sys.stderr)
        return model

    except Exception as e:
        error_msg = f"Wystąpił błąd podczas ładowania modelu z DigitalOcean Spaces: {e}"
        st.error(error_msg)
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return None


# --- Modele Pydantic do strukturyzacji danych ---
class RunStats(BaseModel):
    """
    Model danych do zbierania statystyk biegowych użytkownika,
    w tym wieku, płci, czasu ukończenia biegu na 5 km.
    """
    Wiek: Annotated[int, Field(
        ge=1,
        le=120,
        description="Wiek uczestnika biegu w latach. Musi być liczbą całkowitą z zakresu od 1 do 120.",
        example=30
    )]
    Płeć: Literal["M", "K"] = Field(
        ...,
        description="Płeć uczestnika biegu. Dozwolone wartości to 'M' (Mężczyzna), 'K' (Kobieta).",
        example="K"
    )
    czas_5km: Annotated[Optional[float], Field( # Użycie Optional[float]
        ge=0.0,
        description="Czas ukończenia biegu na dystansie 5 km podany w sekundach. Musi być liczbą nieujemną. Może być opcjonalny.",
        example=1500.5
    )] = None # Ustawienie wartości domyślnej na None

    class Config:
        schema_extra = {
            "example": {
                "Wiek": 25,
                "Płeć": "M",
                "5 km Czas": 1200.0
            }
        }

@observe()
def extract_run_data(user_input: str) -> dict | None:
    """
    Funkcja, która pobiera tekst od użytkownika, używa modelu OpenAI z instructorem
    do ekstrakcji danych zgodnych z modelem RunStats i zwraca je w postaci słownika.

    Args:
        user_input (str): Tekst wprowadzony przez użytkownika (np. "Jestem mężczyzną, mój rocznik to 1999, 5km zrobiłem w 25 minut").

    Returns:
        dict | None: Słownik zawierający wyekstrahowane dane lub None w przypadku błędu walidacji.
    """
    if not instructor_openai_client:
        st.error("Klient OpenAI nie został zainicjowany. Sprawdź konfigurację klucza API.")
        return None

    if not user_input.strip():
        st.warning("Pole tekstowe jest puste. Wprowadź dane, aby kontynuować.")
        return None

    try:
        system_prompt = (
            "Jesteś asystentem, który precyzyjnie ekstrahuje statystyki biegowe z tekstu użytkownika. "
            "Skonwertuj czas tylko dla 5km na sekundy (np. '5 minut' to 300 sekund, '1h 5m' to 3900 sekund). "
            "Jeśli podana wartość czasu jest dla dystansu innego jak 5km zwróć wartość None"
            "Jeśli rocznik jest podany, oblicz wiek na podstawie bieżącego roku. "
            "Pamiętaj, że bieżący rok to 2025." # Pamiętaj, aby aktualizować ten rok, gdy nadejdzie 2026.
            "Upewnij się, że wszystkie wartości są zgodne ze schematem RunStats. "
        )

        extracted_data: RunStats = instructor_openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            response_model=RunStats,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
        )
        return extracted_data.model_dump()
    except Exception as e:
        st.error(f"Wystąpił błąd podczas ekstrakcji danych: {e}")
        return None

# --- Callback dla przycisku 'Wyczyść wyniki' ---
def clear_results_callback():
    """
    Funkcja wywoływana po kliknięciu przycisku 'Wyczyść wyniki'.
    Ustawia stan sesji, aby wyczyścić pole tekstowe i ukryć wyniki.
    """
    st.session_state["user_input_text_area"] = ""
    st.session_state["show_results"] = False
    # NIE wywołujemy st.rerun() tutaj. Streamlit to zrobi automatycznie po callbacku.


# --- Interfejs użytkownika Streamlit ---

st.title("🏃‍♂️ Przewidywanie Czasu Ukończenia Półmaratonu")
st.write("""
Wprowadź swoje dane, aby przewidzieć czas ukończenia półmaratonu.
Podaj płeć, wiek, a także czas na 5km. **Czasy zostaną skonwertowane na sekundy.**
""")

# Załaduj model
pycaret_model = load_pycaret_model_from_do_spaces(
    bucket_name= BUCKET_NAME,
    object_key= Model_pkl_in_spaces
)

# Użyj wartości z session_state jako domyślnej wartości pola tekstowego
# Streamlit automatycznie aktualizuje session_state['user_input_text_area']
# gdy użytkownik wpisze coś w polu.
user_input_value = st.text_area(
    "Podaj swoje dane (np. 'Jestem mężczyzną, mam 25 lat, 5km zrobiłem w 25 minut'):",
    height=100,
    placeholder="Podaj płeć, wiek, czas na 5km.",
    key="user_input_text_area", # Ten klucz jest powiązany z st.session_state["user_input_text_area"]
    value=st.session_state["user_input_text_area"] # Ustawia początkową wartość
)


# --- Przyciski ---
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Oszacuj czas ukończenia biegu"):
        if user_input_value: # Użyj wartości z widżetu, która już jest w session_state
            extracted_dict = extract_run_data(user_input_value)
            if extracted_dict:
                plec_map = {
                    'K': 'Kobieta',
                    'M': 'Mężczyzna'
                }
                plec_skrot = extracted_dict.get('Płeć')
                plec_pelna_nazwa = plec_map.get(plec_skrot, 'nie podano')
                info_lines = [
                    "**Szacowanie czasu na podstawie podanych danych:**",
                    f"- Płeć: {plec_pelna_nazwa}",
                    f"- Wiek: {extracted_dict.get('Wiek', 'nie podano')} lat"
                ]

                if extracted_dict.get('czas_5km') is None:
                    st.info('Podaj Twój czas na 5km')

                if extracted_dict.get('czas_5km') is not None:
                                    total_seconds = extracted_dict.get('czas_5km')
                                    hours = int(total_seconds // 3600)
                                    minutes = int((total_seconds % 3600) // 60)
                                    seconds = int(total_seconds % 60)
                                    info_lines.append(f"- 5km: {hours}h {minutes}min {seconds}s")

                # Wyświetl podsumowanie danych wejściowych i wykonaj predykcję tylko,
                # jeśli wszystkie kluczowe dane są dostępne
                if extracted_dict.get('Płeć') is not None and extracted_dict.get('czas_5km') is not None:
                    st.info("\n".join(info_lines))

                    # Przygotowanie DataFrame dla modelu
                    df = pd.DataFrame([extracted_dict])
                    df = df.rename(columns={
                        'czas_5km': '5 km Czas' # Upewnij się, że nazwa kolumny pasuje do tej, której oczekuje Twój model
                    })

                    # Sprawdź, czy model został załadowany pomyślnie przed użyciem
                    if pycaret_model:
                        predictions = predict_model(pycaret_model, data=df)
                        prediction_seconds = predictions.iloc[0]['prediction_label']
                        prediction_minutes = prediction_seconds / 60

                        hours = int(prediction_seconds // 3600)
                        minutes = int((prediction_seconds % 3600) // 60)
                        seconds = int(prediction_seconds % 60)

                        st.success(f"Przewidywany czas ukończenia półmaratonu: **{hours}h {minutes}m {seconds}s** ({prediction_minutes:.2f} minut)")
                        st.session_state["show_results"] = True
                    else:
                        st.error("Model nie został załadowany. Nie można wykonać predykcji.")
                        st.session_state["show_results"] = False
                else:
                    # To ostrzeżenie pojawi się, jeśli brakuje płci, wieku lub czasu 5km
                    st.warning("Aby oszacować czas, proszę podać płeć, wiek i czas na 5km.")
                    st.session_state["show_results"] = False
            else:
                st.warning("Nie udało się wyodrębnić danych. Spróbuj ponownie lub zmień format tekstu.")
                st.session_state["show_results"] = False
        else:
            st.warning("Proszę wprowadzić dane, aby wyodrębnić informacje.")
            st.session_state["show_results"] = False

with col2:
    # Użycie callbacku `on_click` dla przycisku 'Wyczyść wyniki'
    st.button("Wyczyść wyniki", on_click=clear_results_callback)