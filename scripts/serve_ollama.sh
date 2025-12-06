#!/bin/bash
# ==========================================
# SK-Coder v2.0 - Ollama Inference Server
# ==========================================
# Automatycznie instaluje Ollama, pobiera model z HF i wystawia publiczne API.

set -e

echo "🚀 Rozpoczynam setup serwera inferencyjnego (Ollama)..."

# 1. Instalacja Ollama
if ! command -v ollama &> /dev/null; then
    echo "📦 Instalacja Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✅ Ollama już zainstalowana"
fi

# 2. Uruchomienie serwera w tle
echo "🔄 Uruchamianie serwera Ollama..."
ollama serve > ollama.log 2>&1 &
PID=$!
sleep 5 # Czekaj na start

# 3. Pobranie modelu GGUF z Hugging Face
echo "📥 Pobieranie modelu GGUF 'studiokalmus/sk-coder'..."
# Używamy huggingface-cli do pobrania konkretnego pliku
# Najpierw upewnij się, że HF CLI jest (powinno być po setup_runpod.sh)
if ! command -v huggingface-cli &> /dev/null; then
    pip install huggingface_hub
fi

MODEL_DIR="ollama_models"
mkdir -p $MODEL_DIR

# Pobieranie pliku .gguf (zakładamy, że jest jeden główny lub bierzemy pierwszy)
# Jeśli nie znasz dokładnej nazwy pliku, to jest tricky, ale spróbujemy standardowo
# Zazwyczaj unsloth zapisuje jako 'unsloth.Q4_K_M.gguf' lub podobnie.
# Tutaj pobierzemy wszystkie pliki .gguf z repo (zazwyczaj to ten jeden właściwy)
echo "   Skanowanie repozytorium..."
huggingface-cli download studiokalmus/sk-coder --include "*.gguf" --local-dir $MODEL_DIR --local-dir-use-symlinks False

# Znajdź pobrany plik
GGUF_FILE=$(find $MODEL_DIR -name "*.gguf" | head -n 1)

if [ -z "$GGUF_FILE" ]; then
    echo "❌ BŁĄD: Nie znaleziono pliku .gguf w repozytorium studiokalmus/sk-coder!"
    echo "   Upewnij się, że trening zakończył się sukcesem i plik został wyeksportowany."
    kill $PID
    exit 1
fi

echo "✅ Znaleziono model: $GGUF_FILE"

# 4. Tworzenie modelu w Ollama
echo "🔨 Tworzenie modelu 'sk-coder' w Ollama..."
echo "FROM $GGUF_FILE" > Modelfile
ollama create sk-coder -f Modelfile

# 5. Wystawienie publicznego linku (Cloudflare Tunnel)
echo "🌐 Uruchamianie tunelu publicznego (Cloudflare)..."

# Pobranie cloudflared (jeśli nie ma)
if [ ! -f "cloudflared" ]; then
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    chmod +x cloudflared
fi

# Uruchomienie tunelu do portu 11434
./cloudflared tunnel --url http://localhost:11434 > tunnel.log 2>&1 &
TUNNEL_PID=$!

echo "⏳ Czekam na wygenerowanie linku..."
sleep 8
# Wyciągnij URL z logów
PUBLIC_URL=$(grep -o 'https://.*\.trycloudflare.com' tunnel.log | head -n 1)

echo ""
echo "========================================================"
echo "🎉 SERWER GOTOWY DO PRACY!"
echo "========================================================"
echo "🤖 Model: sk-coder (załadowany w Ollama)"
echo "🔗 OLLAMA URL (Publiczny):"
echo ""
echo "   $PUBLIC_URL"
echo ""
echo "👉 Skopiuj ten link do wtyczki Cline / Continue jako 'Ollama Host'"
echo "========================================================"

# Czekaj w nieskończoność żeby kontenery nie padły
wait $PID
