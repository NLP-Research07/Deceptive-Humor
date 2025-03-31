import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def generate_comment(model, tokenizer, prompt, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7, top_k=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Change to your model path if local
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    fake_claim = "China is Spreading Covid as a biological weapon"
    languages = {
        "Te": "Telugu Script Langauge",
        "En": "English",
        "Hi": "Hindi Script Langauge",
        "Ka": "Kannada Script Langauge",
        "Ta": "Tamil Script Langauge",
        "Te-En": "Telugu-English Code-Mixed Langauge",
        "Hi-En": "Hindi-English Code-Mixed Langauge",
        "Ka-En": "Kannada-English Code-Mixed Langauge",
        "Ta-En": "Tamil-English Code-Mixed Langauge"
    }
    
    results = []
    
    for lang_code, lang_name in languages.items():
        for _ in range(5):  # Generate multiple samples per language
            prompt = f"Generate a satirical and humorous comment about the fake claim: '{fake_claim}' in {lang_name}."
            comment = generate_comment(model, tokenizer, prompt)
            results.append({
                "Language": lang_code,
                "Comment": comment.strip()
            })
    
    with open("llama_DHD_data.txt", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print("Data saved successfully to faux_humor_data.txt")
    
if __name__ == "__main__":
    main()
