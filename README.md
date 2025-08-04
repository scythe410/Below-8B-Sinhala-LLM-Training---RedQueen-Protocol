# RedQueen Llama 3.2 3B - Sinhala Generative QA

**Technical Report:** [Click here for pdf](https://drive.google.com/file/d/1XFPwiwTx5j8yxcBCxmyDZgK5ldpulFw-/view?usp=sharing)
<br>
**GitHub Repo for Scripts and Notebooks:** [Click here](https://github.com/scythe410/Below-8B-Sinhala-LLM-Training---RedQueen-Protocol)

- **Developed by:** [Red Queen Protocol](https://huggingface.co/RedQueenProtocol)
- **Team:** [Ramiru De Silva](https://www.linkedin.com/in/ramirudesilva/), [Senadhi Thimanya](https://www.linkedin.com/in/senadhi-chandrasekara/)
- **Language(s) (NLP):** Sinhala
- **Finetuned from model:** [Llama 3.2 3B IT](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)



This model and LoRA was developed by Ramiru De Silva and Senadhi Thimanya (Team: [RedQueen Protocol](https://huggingface.co/RedQueenProtocol)) for the iCIIT Conclave 2025 Shared Task on Building Compact Sinhala & Tamil LLMs. 
This is a 3-billion parameter, instruction-tuned model that has undergone a novel two-stage fine-tuning process to achieve proficiency in both the Sinhala language and the specific task of generative QA. The entire fine-tuning process was performed efficiently using Low-Rank Adaptation (LoRA) technique.
<br>
The model's creation follows a hierarchical training strategy designed to first build a strong linguistic foundation and then specialize it for a specific task.

### Stage 1: Domain Adaptation (Language Foundation)
The initial model, `RedQueenProtocol/llama-3.2-3b-it-sinhala-rq` (Meta's Llama-3.2-3B-IT copies into a private repo for ease of use), was fine-tuned on the entirety of the Sinhala Wikipedia to create a foundational model with a comprehensive grasp of the language.
- **Dataset:** `RedQueenProtocol/all-articles-from-sinhala-wikipedia-2025-parquet`.
- **Method:** Long articles were tokenized and split into overlapping chunks of 512 tokens to ensure full context was seen during training.
- **Output Model:** The resulting adapter was merged to create the Sinhala domain-expert base model for the next stage: `RedQueenProtocol/sinhala-wiki-2025-LoRA-merged`.

### Stage 2: Task Adaptation (Sequential QA Fine-tuning)
Using the Wikipedia-tuned model as the new base, a single LoRA adapter was sequentially fine-tuned on three distinct QA datasets to progressively accumulate question-answering skills.
<br>
The training sequence was as follows:
1. **Custom Dataset:** Fine-tuned on a manually curated dataset of 528 Sinhala QA pairs (`RedQueenProtocol/sinhala-qna-530-rows`).
2. **Ihalage ELI5 Dataset:** Continued training the same adapter on 10,000 samples from the `ihalage/sinhala-finetune-qa-eli5` dataset.
3. **SiQuAD Dataset:** Performed a final round of training on 13,500 samples from the `janani-rane/SiQuAD` dataset, formatting the inputs as "Context: ... Question: ... Answer: ...".

The **final LoRA adapter**, containing the combined knowledge of all three datasets **and the Wikipedia-tuned base model** was then uploaded here.

## How to Use

You can use this model directly with the `transformers` library pipeline.

```python
import torch
from transformers import pipeline

# Load the model and tokenizer
model_id = "RedQueenProtocol/llama-3.2-3b-it-sinhala-qa-merged"

generator = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

# Prepare a question using the Llama 3 chat template
question = "ශ්‍රී ලංකාවේ උසම කන්ද කුමක්ද?"
prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"

# Generate a response
outputs = generator(
    prompt,
    max_new_tokens=256,
    eos_token_id=generator.tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

# Extract and print the answer
full_response = outputs[0]['generated_text']
answer = full_response.split("<|start_header_id|>assistant<|end_header_id|>\\n\\n")[1].replace("<|eot_id|>", "")

print(f"Question: {question}")
print(f"Answer: {answer.strip()}")