#!/bin/bash
# ==========================================
# SK-Coder v2.0 - RunPod Setup Script
# ==========================================
# Uruchom ten skrypt zaraz po starcie poda, aby przygotować środowisko.

set -e  # Przerwij w przypadku błędu

echo "🚀 Rozpoczynam konfigurację środowiska RunPod..."

# 1. Aktualizacja systemu i instalacja narzędzi
echo "📦 Instalacja narzędzi systemowych..."
apt-get update && apt-get install -y git wget nano htop tmux

# 2. Instalacja zależności Python
echo "🐍 Instalacja zależności Python..."
# Ogranicz liczbę wątków kompilacji, aby nie wysadzić RAMu (częsty problem przy Flash Attn)
export MAX_JOBS=2
pip install --upgrade pip

# Unsloth wymaga specyficznej instalacji, najpierw pytorch
# RunPod zazwyczaj ma już PyTorch, ale sprawdźmy wersję
echo "   Sprawdzanie wersji PyTorch..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Instalacja Unsloth (optymalizacja pod Ampere/Hopper/Volta)
echo "   Instalacja Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Instalacja reszty zależności z requirements.txt
if [ -f "requirements.txt" ]; then
    echo "   Instalacja dependencies z requirements.txt..."
    pip install -r requirements.txt
else
    echo "⚠️ Ostrzeżenie: Nie znaleziono requirements.txt w bieżącym katalogu."
fi

# Instalacja Flash Attention 2 (jeśli dostępne GPU)
echo "⚡ Instalacja Flash Attention 2 (może chwilę potrwać)..."
pip install flash-attn --no-build-isolation

# 3. Konfiguracja Git i Hugging Face
echo "🔧 Konfiguracja Git..."
git config --global credential.helper store

# Sprawdzenie zmiennych środowiskowych
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️ Ostrzeżenie: Brak zmiennej HF_TOKEN. Upewnij się, że dodałeś ją w ustawieniach poda."
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️ Ostrzeżenie: Brak zmiennej WANDB_API_KEY. Logowanie do W&B może się nie udać."
fi

# Logowanie do W&B jeśli jest klucz
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "📈 Logowanie do Weights & Biases..."
    wandb login $WANDB_API_KEY
fi

# Logowanie do HF jeśli jest token
if [ ! -z "$HF_TOKEN" ]; then
    echo "🤗 Logowanie do Hugging Face Hub..."
    huggingface-cli login --token $HF_TOKEN
fi

echo "✅ Środowisko gotowe!"
echo "   Możesz teraz uruchomić trening: python main.py --step all"
