#!/bin/bash

# ========================================
# SK-Coder v2.0 - Setup & Resume Training
# ========================================
# Automatyczny setup środowiska VPS i uruchomienie treningu
# ========================================

set -e  # Exit on error

echo "======================================"
echo "SK-Coder v2.0 - Setup & Resume"
echo "======================================"

# 1. Załaduj secrets
echo "📋 Ładowanie konfiguracji secrets..."
source devops/secrets.sh

# Walidacja
if [[ "$HF_TOKEN" == "WKLEJ_TU_TOKEN_WRITE" ]]; then
    echo "❌ BŁĄD: Musisz najpierw uzupełnić devops/secrets.sh!"
    exit 1
fi

# 2. Instalacja zależności systemowych (jeśli potrzebne)
echo ""
echo "📦 Sprawdzanie zależności systemowych..."
if ! command -v git &> /dev/null; then
    echo "Instalacja git..."
    sudo apt-get update && sudo apt-get install -y git
fi

# 3. Klonowanie repozytorium (jeśli jeszcze nie sklonowane)
REPO_URL="https://github.com/studiokalmus/sk-coder.git"
if [ ! -d ".git" ]; then
    echo ""
    echo "📥 Klonowanie repozytorium..."
    cd ..
    git clone $REPO_URL
    cd sk-coder
else
    echo "✅ Repozytorium już sklonowane"
fi

# 4. Utworzenie środowiska wirtualnego
echo ""
echo "🐍 Konfiguracja środowiska Python..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Środowisko wirtualne utworzone"
else
    echo "✅ Środowisko wirtualne już istnieje"
fi

# Aktywacja venv
source venv/bin/activate

# 5. Instalacja zależności Python
echo ""
echo "📦 Instalacja zależności pip..."
pip install --upgrade pip
pip install -r requirements.txt

# 6. Logowanie do Hugging Face
echo ""
echo "🔐 Logowanie do Hugging Face..."
huggingface-cli login --token $HF_TOKEN

# 7. Logowanie do Weights & Biases
echo ""
echo "🔐 Logowanie do Weights & Biases..."
wandb login $WANDB_API_KEY

# 8. Inicjalizacja repo HuggingFace (jeśli nie istnieje)
echo ""
echo "🔧 Inicjalizacja repozytorium HuggingFace..."
python scripts/init_hf_repo.py

# 9. Sprawdź czy istnieje dataset
if [ ! -f "dataset/training_data.jsonl" ]; then
    echo ""
    echo "⚠️  UWAGA: Brak pliku dataset/training_data.jsonl"
    echo "   Musisz najpierw wygenerować dane:"
    echo "   python scripts/gpu_generate_data.py"
    echo ""
    read -p "Czy wygenerować dane teraz? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python scripts/gpu_generate_data.py
    else
        echo "❌ Przerwano. Wygeneruj dane i uruchom ponownie."
        exit 1
    fi
fi

# 10. Uruchomienie treningu (z auto-resume)
echo ""
echo "======================================"
echo "🚀 URUCHAMIANIE TRENINGU"
echo "======================================"
echo ""
echo "Trening będzie automatycznie wznawiany z ostatniego checkpointu."
echo "Logi dostępne w Weights & Biases: https://wandb.ai"
echo ""

# Uruchom trening z nohup (w tle, przetrwa disconnect SSH)
nohup python scripts/train.py > training.log 2>&1 &
TRAIN_PID=$!

echo "✅ Trening uruchomiony w tle (PID: $TRAIN_PID)"
echo "   Logi: tail -f training.log"
echo ""
echo "Aby zatrzymać trening: kill $TRAIN_PID"
echo ""
echo "======================================"
echo "Setup zakończony pomyślnie!"
echo "======================================"
