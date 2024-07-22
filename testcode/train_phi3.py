# https://github.com/microsoft/Phi-3CookBook/blob/main/code/04.Finetuning/Phi-3-finetune-lora-python.ipynb


# This code block is importing necessary modules and functions for fine-tuning a language model.

# 'randrange' is a function from the 'random' module that generates a random number within the specified range.
from random import randrange

# 'torch' is the PyTorch library, a popular open-source machine learning library for Python.
import torch

# 'load_dataset' is a function from the 'datasets' library by Hugging Face which allows you to load a dataset.
from datasets import load_dataset

# 'LoraConfig' and 'prepare_model_for_kbit_training' are from the 'peft' library. 
# 'LoraConfig' is used to configure the LoRA (Learning from Random Architecture) model.
# 'prepare_model_for_kbit_training' is a function that prepares a model for k-bit training.
# 'TaskType' contains differenct types of tasks supported by PEFT
# 'PeftModel' base model class for specifying the base Transformer model and configuration to apply a PEFT method to.
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType, PeftModel

# Several classes and functions are imported from the 'transformers' library by Hugging Face.
# 'AutoModelForCausalLM' is a class that provides a generic transformer model for causal language modeling.
# 'AutoTokenizer' is a class that provides a generic tokenizer class.
# 'BitsAndBytesConfig' is a class for configuring the Bits and Bytes optimizer.
# 'TrainingArguments' is a class that defines the arguments used for training a model.
# 'set_seed' is a function that sets the seed for generating random numbers.
# 'pipeline' is a function that creates a pipeline that can process data and make predictions.
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    pipeline
)

# 'SFTTrainer' is a class from the 'trl' library that provides a trainer for soft fine-tuning.
from trl import SFTTrainer

# This code block is setting up the configuration for fine-tuning a language model.

# 'model_id' and 'model_name' are the identifiers for the pre-trained model that you want to fine-tune. 
# In this case, it's the 'Phi-3-mini-4k-instruct' model from Microsoft.
# Model Names 
# microsoft/Phi-3-mini-4k-instruct
# microsoft/Phi-3-mini-128k-instruct
# microsoft/Phi-3-small-8k-instruct
# microsoft/Phi-3-small-128k-instruct
# microsoft/Phi-3-medium-4k-instruct
# microsoft/Phi-3-medium-128k-instruct
# microsoft/Phi-3-vision-128k-instruct
# microsoft/Phi-3-mini-4k-instruct-onnx
# microsoft/Phi-3-mini-4k-instruct-onnx-web
# microsoft/Phi-3-mini-128k-instruct-onnx
# microsoft/Phi-3-small-8k-instruct-onnx-cuda
# microsoft/Phi-3-small-128k-instruct-onnx-cuda
# microsoft/Phi-3-medium-4k-instruct-onnx-cpu
# microsoft/Phi-3-medium-4k-instruct-onnx-cuda
# microsoft/Phi-3-medium-4k-instruct-onnx-directml
# microsoft/Phi-3-medium-128k-instruct-onnx-cpu
# microsoft/Phi-3-medium-128k-instruct-onnx-cuda
# microsoft/Phi-3-medium-128k-instruct-onnx-directml
# microsoft/Phi-3-mini-4k-instruct-gguf

model_id = "microsoft/Phi-3-mini-4k-instruct"
model_name = "microsoft/Phi-3-mini-4k-instruct"

# 'dataset_name' is the identifier for the dataset that you want to use for fine-tuning. 
# In this case, it's the 'python_code_instructions_18k_alpaca' dataset from iamtarun (Ex: iamtarun/python_code_instructions_18k_alpaca).
# Update Dataset Name to your dataset name
dataset_name = "iamtarun/python_code_instructions_18k_alpaca"

# 'dataset_split' is the split of the dataset that you want to use for training. 
# In this case, it's the 'train' split.
dataset_split= "train"

# 'new_model' is the name that you want to give to the fine-tuned model.
new_model = "Name of your new model"

# 'hf_model_repo' is the repository on the Hugging Face Model Hub where the fine-tuned model will be saved. Update UserName to your Hugging Face Username
hf_model_repo="UserName/"+new_model

# 'device_map' is a dictionary that maps the model to the GPU device. 
# In this case, the entire model is loaded on GPU 0.
device_map = {"": "cpu"}

# The following are parameters for the LoRA (Learning from Random Architecture) model.

# 'lora_r' is the dimension of the LoRA attention.
lora_r = 16

# 'lora_alpha' is the alpha parameter for LoRA scaling.
lora_alpha = 16

# 'lora_dropout' is the dropout probability for LoRA layers.
lora_dropout = 0.05

# 'target_modules' is a list of the modules in the model that will be replaced with LoRA layers.
target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]

# 'set_seed' is a function that sets the seed for generating random numbers, 
# which is used for reproducibility of the results.
set_seed(1234)



# This code block is used to load a dataset from the Hugging Face Dataset Hub, print its size, and show a random example from the dataset.

# 'load_dataset' is a function from the 'datasets' library that loads a dataset from the Hugging Face Dataset Hub.
# 'dataset_name' is the name of the dataset to load, and 'dataset_split' is the split of the dataset to load (e.g., 'train', 'test').
dataset = load_dataset(dataset_name, split=dataset_split)

# The 'len' function is used to get the size of the dataset, which is then printed.
print(f"dataset size: {len(dataset)}")

# 'randrange' is a function from the 'random' module that generates a random number within the specified range.
# Here it's used to select a random example from the dataset, which is then printed.
print(dataset[randrange(len(dataset))])


# This code block is used to load a tokenizer from the Hugging Face Model Hub.

# 'tokenizer_id' is set to the 'model_id', which is the identifier for the pre-trained model.
# This assumes that the tokenizer associated with the model has the same identifier as the model.
tokenizer_id = model_id

# 'AutoTokenizer.from_pretrained' is a method that loads a tokenizer from the Hugging Face Model Hub.
# 'tokenizer_id' is passed as an argument to specify which tokenizer to load.
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

# 'tokenizer.padding_side' is a property that specifies which side to pad when the input sequence is shorter than the maximum sequence length.
# Setting it to 'right' means that padding tokens will be added to the right (end) of the sequence.
# This is done to prevent warnings that can occur when the padding side is not explicitly set.
tokenizer.padding_side = 'right'


# This code block defines two functions that are used to format the dataset for training a chat model.

# 'create_message_column' is a function that takes a row from the dataset and returns a dictionary 
# with a 'messages' key and a list of 'user' and 'assistant' messages as its value.
def create_message_column(row):
    # Initialize an empty list to store the messages.
    messages = []
    
    # Create a 'user' message dictionary with 'content' and 'role' keys.
    user = {
        "content": f"{row['instruction']}\n Input: {row['input']}",
        "role": "user"
    }
    
    # Append the 'user' message to the 'messages' list.
    messages.append(user)
    
    # Create an 'assistant' message dictionary with 'content' and 'role' keys.
    assistant = {
        "content": f"{row['output']}",
        "role": "assistant"
    }
    
    # Append the 'assistant' message to the 'messages' list.
    messages.append(assistant)
    
    # Return a dictionary with a 'messages' key and the 'messages' list as its value.
    return {"messages": messages}

# 'format_dataset_chatml' is a function that takes a row from the dataset and returns a dictionary 
# with a 'text' key and a string of formatted chat messages as its value.
def format_dataset_chatml(row):
    # 'tokenizer.apply_chat_template' is a method that formats a list of chat messages into a single string.
    # 'add_generation_prompt' is set to False to not add a generation prompt at the end of the string.
    # 'tokenize' is set to False to return a string instead of a list of tokens.
    return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)}


# This code block is used to prepare the 'dataset' for training a chat model.

# 'dataset.map' is a method that applies a function to each example in the 'dataset'.
# 'create_message_column' is a function that formats each example into a 'messages' format suitable for a chat model.
# The result is a new 'dataset_chatml' with the formatted examples.
dataset_chatml = dataset.map(create_message_column)

# 'dataset_chatml.map' is a method that applies a function to each example in the 'dataset_chatml'.
# 'format_dataset_chatml' is a function that further formats each example into a single string of chat messages.
# The result is an updated 'dataset_chatml' with the further formatted examples.
dataset_chatml = dataset_chatml.map(format_dataset_chatml)


# This line of code is used to access and display the first example from the 'dataset_chatml'.

# 'dataset_chatml[0]' uses indexing to access the first example in the 'dataset_chatml'.
# In Python, indexing starts at 0, so 'dataset_chatml[0]' refers to the first example.
# The result is a dictionary with a 'text' key and a string of formatted chat messages as its value.
dataset_chatml[0]


# This code block is used to split the 'dataset_chatml' into training and testing sets.

# 'dataset_chatml.train_test_split' is a method that splits the 'dataset_chatml' into a training set and a testing set.
# 'test_size' is a parameter that specifies the proportion of the 'dataset_chatml' to include in the testing set. Here it's set to 0.05, meaning that 5% of the 'dataset_chatml' will be included in the testing set.
# 'seed' is a parameter that sets the seed for the random number generator. This is used to ensure that the split is reproducible. Here it's set to 1234.
dataset_chatml = dataset_chatml.train_test_split(test_size=0.05, seed=1234)

# This line of code is used to display the structure of the 'dataset_chatml' after the split.
# It will typically show information such as the number of rows in the training set and the testing set.
dataset_chatml


# This code block is used to set the compute data type and attention implementation based on whether bfloat16 is supported on the current CUDA device.

# 'torch.cuda.is_bf16_supported()' is a function that checks if bfloat16 is supported on the current CUDA device.
# If bfloat16 is supported, 'compute_dtype' is set to 'torch.bfloat16' and 'attn_implementation' is set to 'flash_attention_2'.

compute_dtype = torch.float16
attn_implementation = 'sdpa'
attn_implementation = 'eager' # TODO

# This line of code is used to print the value of 'attn_implementation', which indicates the chosen attention implementation.
print(attn_implementation)
    
# This code block is used to load a pre-trained model and its associated tokenizer from the Hugging Face Model Hub.

# 'model_name' is set to the identifier of the pre-trained model.
model_name = "microsoft/Phi-3-mini-4k-instruct"

# 'AutoTokenizer.from_pretrained' is a method that loads a tokenizer from the Hugging Face Model Hub.
# 'model_id' is passed as an argument to specify which tokenizer to load.
# 'trust_remote_code' is set to True to trust the remote code in the tokenizer files.
# 'add_eos_token' is set to True to add an end-of-sentence token to the tokenizer.
# 'use_fast' is set to True to use the fast version of the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, add_eos_token=True, use_fast=True)

# The padding token is set to the unknown token.
tokenizer.pad_token = tokenizer.unk_token

# The ID of the padding token is set to the ID of the unknown token.
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

# The padding side is set to 'left', meaning that padding tokens will be added to the left (start) of the sequence.
tokenizer.padding_side = 'left'

# 'AutoModelForCausalLM.from_pretrained' is a method that loads a pre-trained model for causal language modeling from the Hugging Face Model Hub.
# 'model_id' is passed as an argument to specify which model to load.
# 'torch_dtype' is set to the compute data type determined earlier.
# 'trust_remote_code' is set to True to trust the remote code in the model files.
# 'device_map' is passed as an argument to specify the device mapping for distributed training.
# 'attn_implementation' is set to the attention implementation determined earlier.
model = AutoModelForCausalLM.from_pretrained(
          model_id, torch_dtype=compute_dtype, trust_remote_code=True, device_map="auto",
          attn_implementation=attn_implementation
)


# This code block is used to define the training arguments for the model.

# 'TrainingArguments' is a class that holds the arguments for training a model.
# 'output_dir' is the directory where the model and its checkpoints will be saved.
# 'evaluation_strategy' is set to "steps", meaning that evaluation will be performed after a certain number of training steps.
# 'do_eval' is set to True, meaning that evaluation will be performed.
# 'optim' is set to "adamw_torch", meaning that the AdamW optimizer from PyTorch will be used.
# 'per_device_train_batch_size' and 'per_device_eval_batch_size' are set to 8, meaning that the batch size for training and evaluation will be 8 per device.
# 'gradient_accumulation_steps' is set to 4, meaning that gradients will be accumulated over 4 steps before performing a backward/update pass.
# 'log_level' is set to "debug", meaning that all log messages will be printed.
# 'save_strategy' is set to "epoch", meaning that the model will be saved after each epoch.
# 'logging_steps' is set to 100, meaning that log messages will be printed every 100 steps.
# 'learning_rate' is set to 1e-4, which is the learning rate for the optimizer.
# 'fp16' is set to the opposite of whether bfloat16 is supported on the current CUDA device.
# 'bf16' is set to whether bfloat16 is supported on the current CUDA device.
# 'eval_steps' is set to 100, meaning that evaluation will be performed every 100 steps.
# 'num_train_epochs' is set to 3, meaning that the model will be trained for 3 epochs.
# 'warmup_ratio' is set to 0.1, meaning that 10% of the total training steps will be used for the warmup phase.
# 'lr_scheduler_type' is set to "linear", meaning that a linear learning rate scheduler will be used.
# 'report_to' is set to "wandb", meaning that training and evaluation metrics will be reported to Weights & Biases.
# 'seed' is set to 42, which is the seed for the random number generator.

# LoraConfig object is created with the following parameters:
# 'r' (rank of the low-rank approximation) is set to 16,
# 'lora_alpha' (scaling factor) is set to 16,
# 'lora_dropout' dropout probability for Lora layers is set to 0.05,
# 'task_type' (set to TaskType.CAUSAL_LM indicating the task type),
# 'target_modules' (the modules to which LoRA is applied) choosing linear layers except the output layer..


args = TrainingArguments(
        output_dir="./phi-3-mini-LoRA",
        evaluation_strategy="steps",
        do_eval=True,
        optim="adamw_torch",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        log_level="debug",
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=1e-4,
        fp16 = True, # TODO
        bf16 = False, # TODO
        eval_steps=100,
        num_train_epochs=3,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        seed=42,
)

peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
)


# This code block is used to initialize the SFTTrainer, which is used to train the model.

# 'model' is the model that will be trained.
# 'train_dataset' and 'eval_dataset' are the datasets that will be used for training and evaluation, respectively.
# 'peft_config' is the configuration for peft, which is used for instruction tuning.
# 'dataset_text_field' is set to "text", meaning that the 'text' field of the dataset will be used as the input for the model.
# 'max_seq_length' is set to 512, meaning that the maximum length of the sequences that will be fed to the model is 512 tokens.
# 'tokenizer' is the tokenizer that will be used to tokenize the input text.
# 'args' are the training arguments that were defined earlier.

trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_chatml['train'],
        eval_dataset=dataset_chatml['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=args,
)


# This code block is used to train the model and save it locally.

# 'trainer.train()' is a method that starts the training of the model.
# It uses the training dataset, evaluation dataset, and training arguments that were provided when the trainer was initialized.
trainer.train()