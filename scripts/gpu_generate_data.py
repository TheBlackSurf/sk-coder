#!/usr/bin/env python3
"""
SK-Coder v2.0 - Generowanie Danych (Teacher-Student)
====================================================
Wykorzystuje Qwen2.5-Coder-72B-Instruct z vLLM jako "Teacher"
do generowania wysokiej jakości danych treningowych.

Wymaga: 2x GPU (tensor_parallel_size=2)
"""

import json
import os
from typing import List, Dict
from vllm import LLM, SamplingParams

# Konfiguracja
TEACHER_MODEL = "Qwen/Qwen2.5-Coder-72B-Instruct"
OUTPUT_DIR = "dataset"
TENSOR_PARALLEL = 2

# Prompt systemowy - wymusza Django 5, Tailwind, HTMX
SYSTEM_PROMPT = """Jesteś ekspertem w tworzeniu nowoczesnych aplikacji webowych używając:
- Django 5.x (najnowsza wersja)
- Tailwind CSS (utility-first styling)
- HTMX (interaktywność bez JavaScriptu)

Generuj kod zgodnie z najlepszymi praktykami:
1. Django Class-Based Views (CBV) zamiast Function-Based Views
2. Django ORM z optymalizacjami (select_related, prefetch_related)
3. Tailwind CSS - używaj utility classes, nie custom CSS
4. HTMX - dynamiczne ładowanie treści, formularze bez przeładowania
5. Kod musi być production-ready (security, error handling, logging)
6. Komentarze po polsku dla Studio Kalmus
"""

# Przykładowe prompty dla generowania danych
TRAINING_PROMPTS = [
    "Stwórz system autentykacji użytkowników z Django 5 (login, register, logout) + Tailwind UI",
    "Zaimplementuj CRUD dla bloga z paginacją używając Django CBV + HTMX dla live search",
    "Napisz REST API w Django REST Framework z JWT authentication",
    "Stwórz dashboard e-commerce z filtrami produktów (Django + HTMX + Tailwind)",
    "Zaimplementuj system komentarzy z nested replies używając Django + HTMX",
    "Napisz formularz kontaktowy z walidacją Django Forms + HTMX + Tailwind",
    "Stwórz system powiadomień realtime z Django Channels + HTMX",
    "Zaimplementuj upload plików z preview (Django + HTMX + Tailwind)",
    # Dodaj więcej promptów według potrzeb
]


class TeacherStudentDataGenerator:
    """Generator danych treningowych używający modelu Teacher (Qwen 72B)."""
    
    def __init__(self):
        print(f"🚀 Ładowanie modelu Teacher: {TEACHER_MODEL}")
        print(f"   Konfiguracja: tensor_parallel_size={TENSOR_PARALLEL}")
        
        self.llm = LLM(
            model=TEACHER_MODEL,
            tensor_parallel_size=TENSOR_PARALLEL,
            gpu_memory_utilization=0.9,
            trust_remote_code=True
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=4096,
            repetition_penalty=1.1
        )
        
        print("✅ Model załadowany pomyślnie!")
    
    def generate_responses(self, prompts: List[str]) -> List[Dict]:
        """Generuje odpowiedzi od Teacher modelu."""
        print(f"\n📝 Generowanie odpowiedzi dla {len(prompts)} promptów...")
        
        # Formatowanie promptów z system prompt
        formatted_prompts = [
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            for prompt in prompts
        ]
        
        # Generowanie w trybie batch
        outputs = self.llm.generate(formatted_prompts, self.sampling_params)
        
        # Formatowanie do struktury treningowej
        dataset = []
        for prompt, output in zip(prompts, outputs):
            dataset.append({
                "instruction": prompt,
                "input": "",
                "output": output.outputs[0].text.strip(),
                "system": SYSTEM_PROMPT
            })
        
        print(f"✅ Wygenerowano {len(dataset)} przykładów treningowych")
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str = "training_data.jsonl"):
        """Zapisuje dataset w formacie JSONL."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"💾 Dataset zapisany: {filepath}")
        print(f"   Rozmiar: {len(dataset)} przykładów")


def main():
    """Główna funkcja generująca dane treningowe."""
    print("=" * 60)
    print("SK-Coder v2.0 - Teacher-Student Data Generation")
    print("=" * 60)
    
    # Inicjalizacja generatora
    generator = TeacherStudentDataGenerator()
    
    # Generowanie danych
    dataset = generator.generate_responses(TRAINING_PROMPTS)
    
    # Zapis do pliku
    generator.save_dataset(dataset)
    
    print("\n✅ Generowanie zakończone!")
    print(f"   Dataset: {OUTPUT_DIR}/training_data.jsonl")
    print("\nKolejny krok: python scripts/train.py")


if __name__ == "__main__":
    main()
