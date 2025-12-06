#!/usr/bin/env python3
"""
SK-Coder v2.0 - Skrypt Treningowy
==================================
Trening Qwen2.5-Coder-32B-Instruct z unsloth (4-bit quantization)
Integracja: Weights & Biases + Hugging Face Hub
"""

import os
import sys
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# Walidacja zmiennych środowiskowych
def validate_env():
    """Sprawdza czy wszystkie wymagane klucze API są ustawione."""
    required_vars = ["HF_TOKEN", "WANDB_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var) or os.getenv(var).startswith("WKLEJ_TU")]
    
    if missing:
        print("❌ BŁĄD: Brakujące zmienne środowiskowe:")
        for var in missing:
            print(f"   - {var}")
        print("\n   Uruchom: source devops/secrets.sh")
        sys.exit(1)
    print("✅ Zmienne środowiskowe poprawnie ustawione")

# Konfiguracja
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True

# LoRA Config
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Config
OUTPUT_DIR = "./checkpoints"
HF_REPO = "studiokalmus/sk-coder"
WANDB_PROJECT = "sk-coder-v2"
WANDB_RUN_NAME = "sk-coder-run-v1"


def load_model_and_tokenizer():
    """Ładuje model i tokenizer z unsloth (4-bit)."""
    print(f"🚀 Ładowanie modelu: {MODEL_NAME}")
    print(f"   Konfiguracja: 4-bit quantization, max_seq_length={MAX_SEQ_LENGTH}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    # Dodaj LoRA adaptery
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    print("✅ Model załadowany z LoRA adapterami")
    return model, tokenizer


def load_training_data():
    """Ładuje dataset z formatu JSONL."""
    dataset_path = "dataset/training_data.jsonl"
    
    if not os.path.exists(dataset_path):
        print(f"❌ BŁĄD: Brak pliku {dataset_path}")
        print("   Uruchom najpierw: python scripts/gpu_generate_data.py")
        sys.exit(1)
    
    print(f"📥 Ładowanie datasetu: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"✅ Załadowano {len(dataset)} przykładów treningowych")
    
    return dataset


def format_prompts(examples):
    """Formatuje prompty do formatu Qwen chat."""
    texts = []
    for instruction, input_text, output, system in zip(
        examples["instruction"],
        examples["input"],
        examples["output"],
        examples["system"]
    ):
        text = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}"
        )
        if input_text:
            text += f"\n{input_text}"
        text += f"<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        texts.append(text)
    
    return {"text": texts}


def get_training_args(resume_from_checkpoint=None):
    """Tworzy konfigurację TrainingArguments."""
    return TrainingArguments(
        # Wyjście
        output_dir=OUTPUT_DIR,
        run_name=WANDB_RUN_NAME,
        
        # Weights & Biases (OBOWIĄZKOWE)
        report_to="wandb",
        
        # Hugging Face Hub
        push_to_hub=True,
        hub_model_id=HF_REPO,
        hub_strategy="checkpoint",
        hub_token=os.getenv("HF_TOKEN"),
        
        # Training parameters
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        
        # Optymalizacje
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        
        # Wznawianie
        resume_from_checkpoint=resume_from_checkpoint,
    )


def main():
    """Główna funkcja treningowa."""
    print("=" * 60)
    print("SK-Coder v2.0 - Training Pipeline")
    print("=" * 60)
    
    # Walidacja
    validate_env()
    
    # Ładowanie
    model, tokenizer = load_model_and_tokenizer()
    dataset = load_training_data()
    
    # Formatowanie
    print("\n📝 Formatowanie promptów...")
    dataset = dataset.map(format_prompts, batched=True)
    
    # Sprawdź czy istnieją checkpointy do wznowienia (Lokalnie LUB na HuggingFace)
    resume_checkpoint = None
    
    # 1. Sprawdź lokalnie
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            resume_checkpoint = os.path.join(OUTPUT_DIR, latest)
            print(f"🔄 Znaleziono LOKALNY checkpoint: {resume_checkpoint}")

    # 2. Jeśli brak lokalnego, sprawdź HuggingFace Hub (Smart Resume)
    if not resume_checkpoint:
        print(f"🔍 Sprawdzanie checkpointów na Hugging Hub ({HF_REPO})...")
        try:
            from huggingface_hub import list_repo_files, snapshot_download
            files = list_repo_files(repo_id=HF_REPO, token=os.getenv("HF_TOKEN"))
            # Szukamy folderów checkpoint-X
            hf_checkpoints = [f for f in files if "checkpoint-" in f]
            
            if hf_checkpoints:
                # Znajdź najwyższy numer checkpointu (wyciągamy z ścieżek np. checkpoint-100/config.json)
                import re
                checkpoint_nums = []
                for f in hf_checkpoints:
                    match = re.search(r"checkpoint-(\d+)", f)
                    if match:
                        checkpoint_nums.append(int(match.group(1)))
                
                if checkpoint_nums:
                    latest_step = max(checkpoint_nums)
                    checkpoint_name = f"checkpoint-{latest_step}"
                    print(f"☁️ Znaleziono ZDALNY checkpoint: {checkpoint_name}")
                    print("📥 Pobieranie ostaniego checkpointu z HF (może chwilę potrwać)...")
                    
                    # Pobierz tylko ten folder checkpointu
                    snapshot_download(
                        repo_id=HF_REPO,
                        allow_patterns=[f"{checkpoint_name}/*"],
                        local_dir=OUTPUT_DIR,
                        token=os.getenv("HF_TOKEN")
                    )
                    resume_checkpoint = os.path.join(OUTPUT_DIR, checkpoint_name)
                    print(f"✅ Pobrano i wznowiono z: {resume_checkpoint}")
        except Exception as e:
            print(f"⚠️ Nie udało się sprawdzić/pobrać zdalnych checkpointów: {e}")
            print("   Rozpoczynam trening od zera.")
    
    # Training arguments
    training_args = get_training_args(resume_from_checkpoint=resume_checkpoint)
    
    # Trainer
    print("\n🎯 Inicjalizacja trenera...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )
    
    # Start treningu
    print("\n🚀 Rozpoczynam trening...")
    print(f"   W&B Project: {WANDB_PROJECT}")
    print(f"   W&B Run: {WANDB_RUN_NAME}")
    print(f"   HF Repo: {HF_REPO}")
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # Zapis finalnego modelu
    # Zapis finalnego modelu (Adaptery)
    print("\n💾 Zapisywanie finalnego modelu (LoRA)...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Push adapterów do HuggingFace Hub
    print(f"\n📤 Wysyłanie adapterów do {HF_REPO}...")
    model.push_to_hub(HF_REPO, token=os.getenv("HF_TOKEN"))
    tokenizer.push_to_hub(HF_REPO, token=os.getenv("HF_TOKEN"))
    
    # --- NOWOŚĆ: Generowanie GGUF dla Ollama ---
    print("\n📦 Konwertowanie do GGUF (dla Ollama)...")
    print("   Format: q4_k_m (zbalansowana jakość/prędkość)")
    
    try:
        model.push_to_hub_gguf(
            HF_REPO, 
            tokenizer, 
            quantization_method = "q4_k_m", 
            token = os.getenv("HF_TOKEN")
        )
        print("✅ GGUF wysłany pomyślnie! Twój model jest gotowy do Ollama.")
    except Exception as e:
        print(f"⚠️ Błąd eksportu GGUF: {e}")
        print("   Możesz spróbować ręcznej konwersji później.")

    print("\n✅ Trening i eksport zakończony sukcesem!")
    print(f"   Model LoRA: https://huggingface.co/{HF_REPO}")
    print(f"   Model GGUF: https://huggingface.co/{HF_REPO}/tree/main (szukaj pliku .gguf)")


if __name__ == "__main__":
    main()
