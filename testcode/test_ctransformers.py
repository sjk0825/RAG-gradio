from ctransformers import AutoModelForCausalLM
from transformers import pipeline, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "PawanKrd/Llama-3-8B-Instruct-GGUF", 
    model_file="llama-3-8b-instruct.Q4_K_M.gguf",
    model_type="llama",
    gpu_layers=0,
    hf=True)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe("AI is going to", max_new_tokens=256))
raise Exception




from ctransformers import AutoModelForCausalLM
# ctrnasfomer llama 3 update problem
# https://github.com/marella/ctransformers/issues/210
# https://github.com/gjwgit/llama2/issues/1

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained(
    "QuantFactory/Meta-Llama-3-8B-GGUF", 
    model_file="Meta-Llama-3-8B.Q4_K_M.gguf", 
    model_type="llama",
    gpu_layers=0)

tokenizer = meta-llama/Meta-Llama-3-8B

print(llm("AI is going to"))

