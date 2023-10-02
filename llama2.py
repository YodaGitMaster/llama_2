# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

torch.cuda.empty_cache()

model = "NousResearch/Nous-Hermes-llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Nous-Hermes-llama-2-7b")

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'list the title of 3 scientific paper anout climate change',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")


# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")