#!/bin/bash

# ========================================
# SK-Coder v2.0 - Deploy Inference Server
# ========================================
# Instalacja Ollama + GGUF model + Ngrok tunnel
# ========================================

set -e  # Exit on error

echo "======================================"
echo "SK-Coder v2.0 - Deploy Server"
echo "======================================"

# 1. Załaduj secrets
echo "📋 Ładowanie konfiguracji secrets..."
source devops/secrets.sh

# Walidacja
if [[ "$NGROK_TOKEN" == "WKLEJ_TU_TOKEN" ]]; then
    echo "❌ BŁĄD: Musisz najpierw uzupełnić NGROK_TOKEN w devops/secrets.sh!"
    exit 1
fi

# 2. Instalacja Ollama
echo ""
echo "📦 Instalacja Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama zainstalowana"
else
    echo "✅ Ollama już zainstalowana ($(ollama --version))"
fi

# 3. Uruchom Ollama service
echo ""
echo "🚀 Uruchamianie Ollama service..."
if ! pgrep -x "ollama" > /dev/null; then
    nohup ollama serve > ollama/ollama.log 2>&1 &
    sleep 5  # Poczekaj na uruchomienie
    echo "✅ Ollama service uruchomiony"
else
    echo "✅ Ollama service już działa"
fi

# 4. Pobierz model GGUF
echo ""
echo "📥 Pobieranie modelu SK-Coder (GGUF)..."
echo "   To może potrwać kilka minut..."

# UWAGA: Zakładam, że model GGUF będzie dostępny jako studiokalmus/sk-coder:latest
# Po wytrenowaniu i konwersji do GGUF, załaduj go do Ollama
MODEL_NAME="studiokalmus/sk-coder:latest"

# Sprawdź czy model już istnieje
if ollama list | grep -q "sk-coder"; then
    echo "✅ Model już pobrany"
else
    # Dla nowego modelu - najpierw trzeba go skonwertować do GGUF
    # i dodać do Ollama registry lub załadować lokalnie
    echo "📝 UWAGA: Model musi być najpierw skonwertowany do formatu GGUF"
    echo "   Użyj: llama.cpp/convert.py --outfile sk-coder.gguf"
    echo "   Następnie: ollama create sk-coder -f Modelfile"
    echo ""
    echo "   Przykładowy Modelfile (ollama/Modelfile):"
    echo "   FROM ./sk-coder.gguf"
    echo "   PARAMETER temperature 0.7"
    echo "   PARAMETER top_p 0.95"
    echo "   SYSTEM \"Jesteś SK-Coder - ekspert Django 5 + Tailwind + HTMX\""
    echo ""
fi

# 5. Instalacja Ngrok
echo ""
echo "📦 Instalacja Ngrok..."
if ! command -v ngrok &> /dev/null; then
    # Dla Ubuntu/Debian
    curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | \
        sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && \
        echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | \
        sudo tee /etc/apt/sources.list.d/ngrok.list && \
        sudo apt update && sudo apt install ngrok
    echo "✅ Ngrok zainstalowany"
else
    echo "✅ Ngrok już zainstalowany ($(ngrok version))"
fi

# 6. Konfiguracja Ngrok authtoken
echo ""
echo "🔐 Konfiguracja Ngrok authtoken..."
ngrok config add-authtoken $NGROK_TOKEN
echo "✅ Authtoken skonfigurowany"

# 7. Uruchom Ngrok tunnel
echo ""
echo "🌐 Uruchamianie Ngrok tunnel..."
echo "   Tunel: http://localhost:11434 (Ollama API)"

# Uruchom ngrok w tle
nohup ngrok http 11434 --log=ollama/ngrok.log > ollama/ngrok_output.log 2>&1 &
NGROK_PID=$!

sleep 3  # Poczekaj na uruchomienie

echo "✅ Ngrok tunnel uruchomiony (PID: $NGROK_PID)"
echo ""

# 8. Pobierz publiczny URL
echo "📡 Pobieranie publicznego URL..."
sleep 2
PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*' | grep -o 'https://[^"]*' | head -n1)

if [ -z "$PUBLIC_URL" ]; then
    echo "⚠️  Nie można pobrać URL automatycznie"
    echo "   Sprawdź: http://localhost:4040"
else
    echo ""
    echo "======================================"
    echo "✅ PUBLICZNY URL:"
    echo "   $PUBLIC_URL"
    echo "======================================"
    echo ""
    echo "Przykładowe użycie (cURL):"
    echo "curl $PUBLIC_URL/api/generate -d '{"
    echo "  \"model\": \"sk-coder\","
    echo "  \"prompt\": \"Stwórz formularz logowania Django + Tailwind\""
    echo "}'"
    echo ""
fi

# 9. Zapis konfiguracji
echo "$PUBLIC_URL" > ollama/public_url.txt
echo "💾 URL zapisany w: ollama/public_url.txt"

echo ""
echo "======================================"
echo "Deployment zakończony!"
echo "======================================"
echo ""
echo "Usługi uruchomione:"
echo "  - Ollama: http://localhost:11434"
echo "  - Ngrok Dashboard: http://localhost:4040"
echo "  - Publiczny URL: $PUBLIC_URL"
echo ""
echo "Logi:"
echo "  - tail -f ollama/ollama.log"
echo "  - tail -f ollama/ngrok.log"
echo ""
echo "Aby zatrzymać:"
echo "  - Ollama: pkill ollama"
echo "  - Ngrok: kill $NGROK_PID"
echo ""
