<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Tuning Gemma Model

The Gemma models are text-to-text, decoder-only Large Language Models (LLM) designed and trained by Google.

This example shows how to fine-tune Gemma model on a TPU using `optimum-tpu` to leverage the TPU processing power and Hugging Face's simple tools to adapt a model to specific needs.

For simplicity, we will use the `gemma-2b` variant.

## Setup your Environment

For this example, a single-host `v5elitepod8` TPU will be enough. To set up a TPU environment with Pytorch XLA, you can check this [Google Cloud guide](https://cloud.google.com/tpu/docs/run-calculation-pytorch) that shows how to do that.

Once you have access to the TPU VM, we can clone the `optimum-tpu` repository containing the required scripts:

```shell
git clone https://github.com/huggingface/optimum-tpu.git
cd optimum-tpu
```

Since we are going to use the gated `gemma` model, you will need to log in to your [Hugging Face token](https://huggingface.co/settings/tokens):

```shell
pip install huggingface transformers tokenizers
huggingface-cli login --token YOUR_TOKEN
```

# Load and Prepare the Dataset

We will use [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k), an open source dataset of instruction-following records on categories outlined in the [InstructGPT](https://arxiv.org/abs/2203.02155) paper, including brainstorming, classification, closed QA, generation, information extraction, open QA, and summarization.

We will load the dataset from the hub:

```python
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
```

We can take a look at an example:

```python
>>> dataset = dataset[321]
{'instruction': 'When was the 8088 processor released?',
 'context': 'The 8086 (also called iAPX 86) is a 16-bit microprocessor chip designed by Intel between early 1976 and June 8, 1978, when it was released. The Intel 8088, released July 1, 1979, is a slightly modified chip with an external 8-bit data bus (allowing the use of cheaper and fewer supporting ICs),[note 1] and is notable as the processor used in the original IBM PC design.',
 'response': 'The Intel 8088 processor was released July 1, 1979.',
 'category': 'information_extraction'}
```

We will use a function to format the dataset to into a collection of tasks with the given instructions:

```python
def format_dolly(sample):
    instruction = f"### Instruction\n{sample['instruction']}"
    context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
    response = f"### Answer\n{sample['response']}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    return prompt
```

In addition, we also want to pack multiple samples into one sequence to have a more efficient training. This means that we stack multiple samples to one sequence and split them with an EOS Token. This makes the training more efficient. Packing/stacking samples can be done during training or before. We will do it before training to save time. We created a utility method `pack_dataset` that takes a dataset and a packing function and returns a packed dataset.

```python
from transformers import AutoTokenizer

# Hugging Face model id
model_id = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

To pack/stack our dataset we need to first tokenize it and then we can pack it with the `pack_dataset` method. To prepare our dataset we will now:

1. Format our samples using the template method and add an EOS token at the end of each sample.
2. Tokenize our dataset to convert it from text to tokens.
3. Pack our dataset to 2048 tokens.

```python
# Add utils method to path for loading dataset
import sys
sys.path.append("./examples/language-modeling") # make sure you change this to the correct path
from pack_dataset import pack_dataset

# Template dataset to add prompt to each sample
def template_dataset(sample):
    sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}"
    return sample

# Apply prompt template per sample
dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))
# Print random sample
from random import randint
print(dataset[randint(0, len(dataset))]["text"])

# Tokenize dataset
dataset = dataset.map(
    lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(dataset.features)
)

# Chunk dataset. We use 2048 as the maximum length for packing
lm_dataset = pack_dataset(dataset, chunk_length=2048)

# Save dataset to disk so it can be reused
dataset_path = "tokenized_dolly"
lm_dataset.save_to_disk(dataset_path)
```

# Fine Tune the Model


