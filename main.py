#!/usr/bin/env python3
"""
SK-Coder v2.0 - Główny Konsolowy Interfejs (CLI)
================================================
Służy do uruchamiania poszczególnych etapów pipeline'u:
1. Inicjalizacja repozytorium HF
2. Generowanie danych (Teacher-Student)
3. Trening modelu (Unsloth LoRA)

Użycie:
  python main.py --step init
  python main.py --step data
  python main.py --step train
  python main.py --step all
"""

import argparse
import sys
import subprocess
import os

def run_script(script_path):
    """Uruchamia skrypt pythonowy i czeka na jego zakończenie."""
    if not os.path.exists(script_path):
        print(f"❌ Błąd: Nie znaleziono skryptu {script_path}")
        return False
        
    print(f"\n🚀 Uruchamianie: {script_path}...")
    try:
        # Używamy sys.executable, aby użyć tego samego interpretera
        result = subprocess.run([sys.executable, script_path], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Błąd podczas wykonywania {script_path}: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Przerwano przez użytkownika.")
        return False

def main():
    parser = argparse.ArgumentParser(description="SK-Coder v2.0 Pipeline Manager")
    parser.add_argument(
        "--step", 
        type=str, 
        choices=["init", "data", "train", "all", "serve"], 
        required=True,
        help="Wybierz krok: init (repo), data (generowanie), train (trening), serve (ollama), all (bez serve)"
    )
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(base_dir, "scripts")
    
    # Krok 1: Inicjalizacja Repo
    if args.step in ["init", "all"]:
        print("\n=== KROK 1: Inicjalizacja Repozytorium Hugging Face ===")
        if not run_script(os.path.join(scripts_dir, "init_hf_repo.py")):
            sys.exit(1)
            
    # Krok 2: Generowanie Danych
    if args.step in ["data", "all"]:
        print("\n=== KROK 2: Generowanie Danych Treningowych (Teacher-Student) ===")
        # Sprawdzenie GPU przed uruchomieniem
        import torch
        if torch.cuda.device_count() < 2:
            print("⚠️ OSTRZEŻENIE: Wykryto mniej niż 2 GPU.")
            print("   Skrypt 'gpu_generate_data.py' wymaga 2 GPU (tensor_parallel_size=2).")
            print("   Jeśli masz 1 GPU, edytuj scripts/gpu_generate_data.py i ustaw TENSOR_PARALLEL = 1")
            response = input("   Czy chcesz kontynuować mimo to? (t/n): ")
            if response.lower() not in ['t', 'y', 'tak']:
                sys.exit(1)
                
        if not run_script(os.path.join(scripts_dir, "gpu_generate_data.py")):
            sys.exit(1)

    # Krok 3: Trening
    if args.step in ["train", "all"]:
        print("\n=== KROK 3: Trening Modelu (Unsloth + LoRA) ===")
        if not run_script(os.path.join(scripts_dir, "train.py")):
            sys.exit(1)
            
    # Krok 4: Serwowanie (Ollama)
    if args.step == "serve":
        print("\n=== KROK 4: Uruchamianie Serwera Inferencyjnego (Ollama) ===")
        # Skrypt bashowy wymaga uruchomienia przez bash
        serve_script = os.path.join(scripts_dir, "serve_ollama.sh")
        os.chmod(serve_script, 0o755)
        try:
            subprocess.run([serve_script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Błąd serwera: {e}")
            sys.exit(1)
            
    if args.step == "all":
        print("\n🎉 Cały pipeline zakończony sukcesem!")

if __name__ == "__main__":
    main()
