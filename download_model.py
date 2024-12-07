from huggingface_hub import hf_hub_download

model_repo = "bartowski/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF"
file_name = "Llama-3.1-Nemotron-70B-Instruct-HF-Q5_K_S.gguf"
save_folder = "app/model"

file_path = hf_hub_download(repo_id=model_repo, filename=file_name, local_dir=save_folder)
print(f"GGUF file save to: {file_path}")

