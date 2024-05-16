import os
from random import randint
from functools import partial
from itertools import chain

from datasets import load_dataset
from transformers import AutoTokenizer

os.environ["PJRT_DEVICE"] = "TPU"

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MODEL_ID = "google/gemma-2b"

remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

import sys
sys.path.append("./examples/language-modeling")
from pack_dataset import pack_dataset

def preprocess_dolly15k(dataset_path):
    if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        return
    # Load dataset from the hub
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    def format_dolly(sample):
        instruction = f"### Instruction\n{sample['instruction']}"
        context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
        response = f"### Answer\n{sample['response']}"
        # join all the parts together
        prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
        return prompt

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # template dataset to add prompt to each sample
    def template_dataset(sample):
        sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}"
        return sample

    # apply prompt template per sample
    dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))
    # print random sample
    print(dataset[randint(0, len(dataset))]["text"])

    # tokenize dataset
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(dataset.features)
    )

    # chunk dataset
    lm_dataset = pack_dataset(dataset, chunk_length=2048) # We use 2048 as the maximum length for packing
    # save train_dataset to disk
    lm_dataset.save_to_disk(dataset_path)


def format_dolly(sample):
    instruction = f"### Instruction\n{sample['instruction']}"
    context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
    response = f"### Answer\n{sample['response']}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    return prompt

def main():
    # dataset_path = "tokenized_dolly"
    # preprocess_dolly15k(dataset_path)

    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    # dataset = dataset.train_test_split(test_size=0.2)

    from transformers import AutoTokenizer

    # Hugging Face model id
    model_id = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    import sys
    sys.path.append("./examples/language-modeling") # make sure you change this to the correct path
    from pack_dataset import pack_dataset

    # Template dataset to add prompt to each sample
    def template_dataset(sample):
        sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}"
        return sample
    dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))

    lm_dataset = pack_dataset(dataset, chunk_length=1024)

    import IPython
    IPython.embed()


import torch
from datasets import load_from_disk
from optimum.tpu import AutoModelForCausalLM, Trainer
from transformers import (
    AutoTokenizer,
    default_data_collator,
)

from transformers.training_args import TrainingArguments
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd.xla_sharding as xs
import torch_xla.runtime as xr
from torch_xla.distributed.fsdp.utils import apply_xla_patch_to_nn_linear
from torch_xla.distributed.fsdp import checkpoint_module



def patch_for_spmd(model, spmd_mesh, torch_dtype, spmd_grad_chkpt=True):

    # Replace the linear layer
    model = apply_xla_patch_to_nn_linear(model, xs.xla_patched_nn_linear_forward)

    # Set the dtype, and move to the XLA device when parameters are already initialized
    model = model.to(xm.xla_device(), torch_dtype)

    # Shard each parameter in the model based on the sharding strategy provided.
    for name, param in model.named_parameters():
        # Apply 2D sharding:
        # We don't care about layernorm's weights, and
        # LLaMA/Gemma models don't use biases.
        if len(param.shape) == 1:
            continue

        if 'embed_tokens' in name:
            xs.mark_sharding(param, spmd_mesh, ('model', 'data'))
        elif 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
            xs.mark_sharding(param, spmd_mesh, ('data', 'model'))
        elif 'o_proj' in name:
            xs.mark_sharding(param, spmd_mesh, ('model', 'data'))
        elif 'gate_proj' in name or 'up_proj' in name:
            xs.mark_sharding(param, spmd_mesh, ('model', 'data'))
        elif 'down_proj' in name:
            xs.mark_sharding(param, spmd_mesh, ('data', 'model'))
        elif 'lm_head' in name:
            xs.mark_sharding(param, spmd_mesh, ('model', 'data'))
        else:
            continue
        print('> [2D] Sharding tensor', name, param.shape)
        print(f'{name} {torch_xla._XLAC._get_xla_sharding_spec(param)}')

    for i, _block in enumerate(model.model.layers):
        # LLaMA/Gemma specific
        xs.apply_backward_optimization_barrier(model.model.layers[i])

    if spmd_grad_chkpt:
        print("Applying gradient checkpointing")
        for i, block in enumerate(model.model.layers):
            # LLaMA/Gemma specific
            model.model.layers[i] = checkpoint_module(block)

def train(model_id, dataset_path):
    torch_dtype = torch.bfloat16

    # Enable SPMD execution mode
    xr.use_spmd()
    # Place DCN on an independent axis in the mesh. Model parameters should be
    # replicated along the DCN axis, and inputs and activations should have
    # the batch dimension sharded along the combined DCN and data axes.
    num_devices = xr.global_runtime_device_count()
    # model_axis = 1
    # dcn_axis = 1
    # data_axis = num_devices // model_axis // dcn_axis
    # mesh data setup
    ici_mesh_shape = (1, num_devices, 1)
    dcn_mesh_shape = (1, 1, 1)
    axis_names=('dcn', 'data', 'model')
    # Note that we do not pass the spmd_mesh to the model because it is not JSON-serializable.
    spmd_mesh = xs.HybridMesh(ici_mesh_shape=ici_mesh_shape, dcn_mesh_shape=dcn_mesh_shape, axis_names=axis_names)

    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        do_train=True,
        output_dir="/tmp/output",
        overwrite_output_dir=True,
        save_strategy="no",
        logging_strategy="no",
        remove_unused_columns=False,
        optim="adafactor",
        dataloader_drop_last=True,
        learning_rate=5e-5,
        max_steps=10,
        logging_steps=10,
    )
    # HACK: We need to pass the spmd_mesh to the model, but it is not JSON-serializable.
    training_args.spmd_mesh=spmd_mesh

    dataset = load_from_disk(dataset_path)

    # load model from the hub with a bnb config
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_cache=False,
        ici_mesh_shape=ici_mesh_shape,
        dcn_mesh_shape=dcn_mesh_shape,
        axis_names=axis_names,
        )

    patch_for_spmd(model, spmd_mesh, torch_dtype=torch_dtype)

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,  # no special collator needed since we stacked the dataset
    )

    # Start training
    trainer.train()

    trainer.save_model()  # Saves the tokenizer too for easy upload


if __name__ == "__main__":
    dataset_path = "tokenized_dolly"
    preprocess_dolly15k(dataset_path=dataset_path)
    train(MODEL_ID, dataset_path)
    # main()