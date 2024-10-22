import transformers
from time import time

def create_text_generation_pipeline(model, tokenizer):
    time_1 = time()
    query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        max_new_tokens=1000,
        device_map="auto",
    )
    time_2 = time()
    print(f"Prepare pipeline: {round(time_2 - time_1, 3)} sec.")
    return query_pipeline
