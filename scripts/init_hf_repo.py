#!/usr/bin/env python3
"""
SK-Coder v2.0 - Inicjalizacja Repozytorium Hugging Face
========================================================
Automatyczne tworzenie publicznego repo 'studiokalmus/sk-coder'

UWAGA: Repo jest PUBLICZNE (private=False) dla łatwego pobierania
bez konieczności logowania na różnych maszynach.
"""

import os
import sys
from huggingface_hub import HfApi, create_repo

def load_hf_token():
    """Ładuje token HF ze zmiennej środowiskowej."""
    token = os.getenv("HF_TOKEN")
    if not token or token == "WKLEJ_TU_TOKEN_WRITE":
        print("❌ BŁĄD: Brak poprawnego HF_TOKEN!")
        print("   Uruchom najpierw: source devops/secrets.sh")
        sys.exit(1)
    return token

def init_repo():
    """Sprawdza/tworzy repozytorium studiokalmus/sk-coder."""
    token = load_hf_token()
    api = HfApi(token=token)
    
    repo_id = "studiokalmus/sk-coder"
    
    try:
        # Sprawdź czy repo już istnieje
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"✅ Repozytorium '{repo_id}' już istnieje.")
        
    except Exception:
        # Utwórz nowe publiczne repo
        print(f"🔧 Tworzenie PUBLICZNEGO repozytorium '{repo_id}'...")
        create_repo(
            repo_id=repo_id,
            private=False,  # PUBLICZNE - łatwe pobieranie bez logowania
            repo_type="model",
            token=token,
            exist_ok=True
        )
        print(f"✅ Repozytorium '{repo_id}' utworzone pomyślnie!")
        print("   Status: PUBLICZNE (private=False)")

if __name__ == "__main__":
    init_repo()
