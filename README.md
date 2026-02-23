# Education Guidance — Undergraduate Admissions Chatbot

> **Domain-specific LLM fine-tuning for educational guidance**  
> Fine-tuned TinyLlama-1.1B · QLoRA (4-bit NF4 + LoRA via PEFT) · Gradio Deployment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Qc4pt-hLTeuxsm4A_eUgCgV9L7OdHJIy?usp=sharing)

## Demo Video Link

(https://youtu.be/I0-Sa7g20Vc)

---

## Project Overview

**EduGuide** is a domain-specific conversational AI assistant fine-tuned to help high school graduates navigate two of the most critical decisions in undergraduate admissions:

1. **Choosing a university major or institution**
2. **Finding scholarships and understanding financial aid**

Millions of students face these decisions every year with limited access to personalised guidance. EduGuide democratises access to high-quality, domain-specific admissions advice through a fine-tuned language model deployed as an interactive chatbot.

| Property | Detail |
|---|---|
| **Base Model** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| **Fine-tuning Method** | QLoRA — 4-bit NF4 quantisation + LoRA adapters (PEFT) |
| **Dataset** | 200 hand-crafted pairs → 1,162 after augmentation |
| **Topics** | Choosing a Major / University · Scholarships & Financial Aid |
| **Training Hardware** | Google Colab Tesla T4 (15.6 GB VRAM) |
| **Deployment** | Gradio ChatInterface (public Colab link) |

---

## Repository Structure

```
eduguide-chatbot/
├── Thierry_SHYAKA_Education_Guide_Chatbot_FineTuning.ipynb   # Main notebook
├── undergraduate_chatbot_dataset_expanded.csv                 # Hand-crafted dataset
└── README.md                                                  # This file
```

---

## Dataset

### Overview

The dataset is entirely **hand-crafted** — every instruction-response pair was written to reflect the real questions high school graduates ask when applying for undergraduate study. No external data sources were used.

| Property | Value |
|---|---|
| **Source** | Hand-crafted by the author |
| **Format** | Instruction-Response pairs (generative QA) |
| **Raw pairs** | 200 unique, matched pairs |
| **Category 1** | Choosing a Major / University — 100 pairs |
| **Category 2** | Scholarships & Financial Aid — 100 pairs |
| **Missing values** | 0 |
| **Duplicate rows** | 0 |
| **Avg instruction length** | 9.0 words |
| **Avg response length** | 34.8 words (range: 24–47) |

### Preprocessing Pipeline

Before training, all data passes through a 5-step quality pipeline:

| Step | Operation | Result |
|---|---|---|
| 1 | **Text Normalisation** | Strip whitespace, remove non-ASCII chars | 200 rows |
| 2 | **Null Removal** | Drop rows with missing values | 200 rows |
| 3 | **Length Filtering** | Remove pairs < 5 words or > 150 words | 200 rows |
| 4 | **Mismatch Detection** | Keyword-based semantic alignment check | 166 rows |
| 5 | **Deduplication** | Drop exact duplicates | 166 rows |

**34 pairs removed** by the mismatch filter — ensuring every remaining pair is semantically aligned within its domain category.

### Data Augmentation

To reach the recommended 1,000–5,000 training examples, **rule-based paraphrasing augmentation** was applied to the 166 clean pairs:

| Technique | Description |
|---|---|
| **Prefix wrapping** | Wraps instructions in conversational openers (`"I'd like to know: ..."`) |
| **Question-word substitution** | Replaces `"How"` → `"Can you explain how"`, `"What"` → `"Could you describe what"`, etc. |
| **Response suffix variation** | Optionally appends encouraging endings to responses |

**Result:** 166 original × 6 augmented variants + 166 originals = **1,162 total pairs**

```
Category split after augmentation:
  Scholarships & Financial Aid     630
  Choosing a Major / University    532
```

### Dataset Split

| Split | Samples | Percentage |
|---|---|---|
| **Training** | 1,045 | 90% |
| **Evaluation** | 117 | 10% |

Stratified sampling ensures both categories are equally represented in both sets.

---

## Fine-Tuning Methodology

### Model: TinyLlama-1.1B-Chat-v1.0

TinyLlama was selected for its balance of capability and efficiency on constrained hardware:

| Criterion | Value |
|---|---|
| Parameters | 1.1 billion |
| Architecture | LLaMA-2 decoder-only transformer |
| VRAM at load (4-bit) | 0.77 GB |
| Peak VRAM during training | 5.18 GB |
| Context window | 2,048 tokens |

### QLoRA Setup

The model is loaded in **4-bit NF4 precision** using `BitsAndBytesConfig`:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

This reduces the memory footprint from ~4.4 GB (fp16) to **0.77 GB**, leaving ample VRAM for LoRA adapters and optimiser states.

### LoRA Configuration

```python
LoraConfig(
    r=16,                          # Rank — Experiment 1 & 2
    lora_alpha=32,                 # Scaling factor (alpha/r = 2.0)
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

| LoRA Parameter | Value |
|---|---|
| Total parameters | 617.9M |
| **Trainable parameters** | **2.25M (0.36%)** |
| Frozen parameters | 615.6M |

### Prompt Template

All pairs are formatted using the **Alpaca instruction-response template**:

```
### System:
You are EduGuide, a friendly and knowledgeable educational assistant that helps
high school graduates apply for undergraduate studies...

### Instruction:
<student question>

### Response:
<EduGuide answer>
```

### Training Configuration

| Setting | Value |
|---|---|
| Optimiser | Paged AdamW 32-bit |
| LR Scheduler | Cosine decay |
| Warmup ratio | 5% |
| Precision | fp16 mixed precision |
| Max sequence length | 512 tokens |
| Checkpoint saving | Every epoch (best kept) |

---

## Hyperparameter Experiments

Three experiments were run to find the optimal configuration:

| Exp | Learning Rate | Batch Size | Grad Accum | Eff. Batch | Epochs | LoRA r | Train Loss | Eval Loss | Perplexity | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | `2e-4` | 2 | 4 | 8 | 3 | 16 | 0.6485 | 0.4277 | 1.53 | 11.1 min |
| **2** | `1e-4` | 2 | 4 | 8 | 3 | 16 | 0.2545 | 0.2193 | 1.25 | 11.2 min |
| **3** | `2e-4` | 4 | 2 | 8 | 2 | 8  | 0.1427 | 0.1150 | **1.12** | **6.9 min** |

**Best experiment: Experiment 3** — larger batch size with fewer epochs and smaller LoRA rank achieved the lowest perplexity (1.12) in the shortest time (6.9 minutes).

**Key finding:** Larger effective batch size outperformed more epochs at small batch size. Experiment 3 trained faster and generalised better.

---

## Performance Metrics

Evaluation was performed on 30 sampled pairs from the held-out evaluation set using standard NLP metrics.

### Fine-tuned Model (Experiment 1)

| Metric | Score | Interpretation |
|---|---|---|
| **BLEU** | 0.2094 | Strong n-gram precision for domain-specific QA |
| **ROUGE-1** | 0.3852 | Good unigram coverage |
| **ROUGE-2** | 0.2693 | Strong phrase-level similarity |
| **ROUGE-L** | 0.3484 | Good structural match |
| **Perplexity** | 1.53 | Excellent — model is confident on in-domain text |

### Base Model vs Fine-tuned Comparison

| Metric | Base Model | Fine-tuned | Improvement |
|---|---|---|---|
| **BLEU** | 0.0159 | 0.2094 | **+1,217%** |
| **ROUGE-1** | 0.2016 | 0.3852 | **+91.1%** |
| **ROUGE-2** | 0.0369 | 0.2693 | **+629.8%** |
| **ROUGE-L** | 0.1238 | 0.3484 | **+181.4%** |

---

## Conversation Examples

### Before Fine-tuning (Base Model)

**Q: How can I choose the right major for me?**
> *"I am unable to suggest or give you specific advice on which majors would best suit your interests and abilities. However, there are many resources available online..."*

Refuses the question. Generic and unhelpful.

---

### After Fine-tuning (EduGuide)

**Q: How can I choose the right major for me?**
> *"Think about your strengths, interests, and long-term career goals. Pick a field that challenges you but still excites you — that's where you'll thrive the most. Don't hesitate to reach out to your school counselor for personalised advice."*

Confident, specific, encouraging, and domain-appropriate.

---

**Q: What documents are needed for a scholarship application?**
> *"You'll need transcripts, a personal statement, recommendation letters, and proof of enrollment or acceptance. Some schemes also ask for a CV, academic portfolio, and financial information. Check the application guidelines carefully before starting."*

Accurate, structured, practical.

---

**Q: What are common mistakes students make when choosing a university?**
> *"Some students pick schools just for prestige or because friends go there. Instead, focus on program quality, accommodation, cost, and career options to ensure your choice fits your long-term goals accurately."*

Directly addresses the question with actionable advice.

---

**Q: [Out-of-domain] What's a good recipe for pasta?**
> *"I'm EduGuide — I specialise in undergraduate admissions advice. I can help you choose a major, pick a university, or find scholarships and financial aid. Try asking me something in those areas!"*

Out-of-domain guard works correctly — politely redirects without breaking.

---

## How to Run

### Prerequisites
- Google account with Google Drive access
- Colab GPU runtime: `Runtime → Change runtime type → T4 GPU`
- Dataset file `undergraduate_chatbot_dataset_expanded.csv` saved to your Google Drive root

### Step-by-step

**1. Open the notebook**

Click the Colab badge at the top of this README, or open `Thierry_SHYAKA_Education_Guide_Chatbot_FineTuning.ipynb` directly in Google Colab.

**2. Run the install cell (Cell 1.1 only)**

```
Run Cell 1.1 → wait for "All libraries installed."
```

**3. Restart the runtime** *(mandatory)*

```
Runtime → Restart session → OK
```

**4. Run from Cell 1.2 downward**

Do **not** re-run Cell 1.1 after restarting. Run all remaining cells in order from top to bottom.

**5. Select your experiment**

In Cell 9.1, set `EXPERIMENT = 1`, `2`, or `3` before running the training section. Run cells 9.1 through 9.5 for each experiment.

**6. Evaluate and deploy**

After training, run Sections 10–12 in order. Section 12 launches the Gradio chatbot with a public URL.

### Session Recovery

If your Colab session disconnects after training, run the **SESSION RECOVERY CELL** (between Sections 9 and 10) before proceeding to evaluation. It reloads the model, tokenizer, and dataset from Google Drive without needing to retrain.

---

## Environment

| Library | Version | Purpose |
|---|---|---|
| `transformers` | 4.44.2 | Model loading, tokenization, training |
| `peft` | 0.12.0 | LoRA adapters (PEFT) |
| `bitsandbytes` | latest | 4-bit NF4 quantisation |
| `accelerate` | 0.33.0 | Mixed-precision training |
| `trl` | 0.9.6 | Supervised fine-tuning utilities |
| `datasets` | 2.20.0 | HuggingFace dataset handling |
| `evaluate` | latest | BLEU and ROUGE metrics |
| `gradio` | latest | Chatbot deployment UI |
| `numpy` | **1.26.4** | Pinned — binary compatibility on Python 3.12 |
| `torch` | 2.10.0+cu128 | GPU training (CUDA 12.8) |

> **Note:** numpy must be pinned to `1.26.4` before all other installs. The install cell handles this automatically.

---

## Key Design Decisions

**Why TinyLlama?** At 1.1B parameters it fits comfortably on Colab's free T4 GPU in 4-bit precision, trains in under 12 minutes per experiment, and still produces domain-coherent responses after fine-tuning.

**Why QLoRA?** Training only 0.36% of parameters (2.25M out of 617.9M) makes the project feasible on free compute while achieving over 1,200% improvement in BLEU score versus the base model.

**Why rule-based augmentation?** External data would introduce noise and potential domain drift. Rule-based paraphrasing preserves the semantic accuracy of every hand-crafted answer while expanding the dataset to the 1,000+ recommended training size.

**Why Alpaca template?** TinyLlama was pre-trained on instruction-following data in this format. Matching the original template maximises the benefit of transfer learning.

---

## Limitations

| Limitation | Notes |
|---|---|
| 166 unique source pairs | Model may repeat phrasing across similar questions |
| Static knowledge | No live data — scholarship details may be outdated |
| English only | Does not support multilingual queries |
| Single-turn only | No persistent conversation memory across sessions |
| Colab-hosted | Public URL expires after one week; no permanent hosting |

---

## Future Work

- Expand to 500+ truly unique source pairs for stronger generalisation
- Add Retrieval-Augmented Generation (RAG) for real-time scholarship lookups
- Deploy permanently to HuggingFace Spaces for persistent public access
- Add multi-turn conversation history management
- Multilingual support for international student queries

---

## Author

**Thierry SHYAKA**  
Domain-Specific LLM Fine-Tuning Project  
Model: TinyLlama-1.1B · Method: QLoRA · Deployment: Gradio

---

## License

This project is for academic and educational purposes.  
Base model: [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) — Apache 2.0 License.
