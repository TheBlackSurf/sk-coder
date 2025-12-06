# SK-Coder v2.0

**Specjalizowany model do generowania kodu Django 5 + Tailwind CSS + HTMX**  
*Studio Kalmus - Professional MLOps Pipeline*

---

## 📋 Opis Projektu

SK-Coder v2.0 to dostosowany model językowy oparty na **Qwen2.5-Coder-32B-Instruct**, wyspecjalizowany w generowaniu wysokiej jakości kodu dla:
- **Django 5.x** (Class-Based Views, ORM, Security)
- **Tailwind CSS** (utility-first styling)
- **HTMX** (interaktywność bez JavaScript frameworks)

### Kluczowe Cechy
✅ Fine-tuned z wykorzystaniem **Teacher-Student** (Qwen 72B → 32B)  
✅ **4-bit quantization** z unsloth (efektywne treningi)  
✅ Integracja **Weights & Biases** (monitoring treningów)  
✅ Automatyczny push do **Hugging Face Hub**  
✅ Deployment z **Ollama + Ngrok** (publiczny API endpoint)  

---

## 🗂️ Struktura Projektu

```
sk-coder/
├── scripts/
│   ├── init_hf_repo.py          # Inicjalizacja HuggingFace repo
│   ├── gpu_generate_data.py     # Generowanie danych (Teacher-Student)
│   └── train.py                 # Pipeline treningowy
├── devops/
│   ├── secrets.sh               # Konfiguracja API keys (GIT-IGNORED)
│   ├── setup_and_resume.sh      # Automatyczny setup VPS + trening
│   └── deploy_server.sh         # Deploy Ollama + Ngrok
├── configs/                     # Dodatkowe konfiguracje
├── dataset/                     # Wygenerowane dane treningowe (GIT-IGNORED)
├── ollama/                      # Logi i pliki Ollama
├── requirements.txt             # Zależności Python
└── README.md                    # Ten plik
```

---

## 🚀 Quickstart

### 1️⃣ Klonowanie Repozytorium

```bash
git clone https://github.com/studiokalmus/sk-coder.git
cd sk-coder
```

### 2️⃣ Konfiguracja Secrets

**⚠️ KRYTYCZNE:** Utwórz plik `devops/secrets.sh` i uzupełnij tokeny:

```bash
cp devops/secrets.sh.example devops/secrets.sh
nano devops/secrets.sh
```

Wypełnij:
- `HF_TOKEN` - Hugging Face token (write access)
- `NGROK_TOKEN` - Ngrok authtoken
- `WANDB_API_KEY` - Weights & Biases API key

### 3️⃣ Automatyczny Setup (VPS)

```bash
chmod +x devops/setup_and_resume.sh
./devops/setup_and_resume.sh
```

Ten skrypt:
- ✅ Instaluje zależności
- ✅ Loguje do HF + W&B
- ✅ Generuje dane (opcjonalnie)
- ✅ Uruchamia trening w tle z auto-resume

---

## 📊 Pipeline Treningowy

### Krok 1: Generowanie Danych (Teacher-Student)

```bash
source devops/secrets.sh
python scripts/gpu_generate_data.py
```

**Wymaga:** 2x GPU (tensor_parallel_size=2)  
**Model Teacher:** Qwen2.5-Coder-72B-Instruct  
**Output:** `dataset/training_data.jsonl`

### Krok 2: Fine-tuning

```bash
python scripts/train.py
```

**Model:** Qwen2.5-Coder-32B-Instruct (4-bit)  
**Metoda:** LoRA (r=16, alpha=16)  
**Monitoring:** Weights & Biases  
**Auto-push:** Hugging Face Hub (`studiokalmus/sk-coder`)

### Obsługa Wznawiania

Trening automatycznie wznawia się z ostatniego checkpointu:
```bash
# Checkpointy w: ./checkpoints/checkpoint-{step}/
python scripts/train.py  # Auto-detects latest checkpoint
```

---

## 🌐 Deployment (Inference Server)

### Deploy z Ollama + Ngrok

```bash
chmod +x devops/deploy_server.sh
./devops/deploy_server.sh
```

Ten skrypt:
1. ✅ Instaluje Ollama
2. ✅ Ładuje model GGUF
3. ✅ Konfiguruje Ngrok tunnel
4. ✅ Zwraca publiczny URL

### Przykładowe Użycie (API)

```bash
# Pobierz publiczny URL
PUBLIC_URL=$(cat ollama/public_url.txt)

# Generuj kod
curl $PUBLIC_URL/api/generate -d '{
  "model": "sk-coder",
  "prompt": "Stwórz system autentykacji Django 5 z Tailwind UI"
}'
```

---

## 📦 Wymagania Systemowe

### Trening (Teacher-Student + Fine-tuning)
- **GPU:** 2x A100 80GB (lub równoważne)
- **RAM:** 128GB+
- **Storage:** 500GB+ SSD
- **CUDA:** 12.1+

### Inference (Ollama)
- **GPU:** 1x RTX 3090 / 4090 (24GB)
- **RAM:** 32GB+
- **Storage:** 100GB SSD

---

## 🔧 Konfiguracja Zaawansowana

### Training Arguments (scripts/train.py)

```python
TrainingArguments(
    num_train_epochs=3,              # Liczba epok
    per_device_train_batch_size=2,   # Batch size
    learning_rate=2e-4,              # Learning rate
    warmup_steps=100,                # Warmup steps
    save_steps=100,                  # Częstotliwość checkpointów
    # ... więcej w pliku
)
```

### LoRA Configuration

```python
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", ...]
```

---

## 📈 Monitoring

### Weights & Biases
- Dashboard: https://wandb.ai/studiokalmus/sk-coder-v2
- Run name: `sk-coder-run-v1`

### Logi Lokalne
```bash
# Trening
tail -f training.log

# Ollama
tail -f ollama/ollama.log

# Ngrok
tail -f ollama/ngrok.log
```

---

## 🛡️ Bezpieczeństwo

### ⚠️ WAŻNE: Secrets Management

- **NIE COMMITUJ** `devops/secrets.sh` do Git!
- Plik jest w `.gitignore` - upewnij się że tam pozostaje
- Na VPS utwórz secrets.sh **ręcznie po sklonowaniu**

### Publiczne Repo
- Repozytorium GitHub: **PUBLICZNE**
- Repozytorium HuggingFace: **PUBLICZNE** (łatwy dostęp)
- Brak kluczy API w kodzie źródłowym ✅

---

## 🤝 Wsparcie

**Studio Kalmus**  
📧 Email: support@studiokalmus.com  
🌐 Website: https://studiokalmus.com

---

## 📝 Licencja

Proprietary - Studio Kalmus © 2024

---

## 🎯 Roadmap

- [ ] Wersja 2.1: Obsługa Django REST Framework
- [ ] Wersja 2.2: Integracja z Celery tasks
- [ ] Wersja 2.3: Deployment templates (Docker, K8s)
- [ ] Wersja 3.0: Multi-modal (code + UI screenshots)

---

**Ostatnia aktualizacja:** 2024-12-06  
**Wersja:** 2.0.0
