import os
import sys
import logging
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from utils import setup_logger  # Make sure this is available in your utils path

# Setup logger
logger = setup_logger("dataloader")

# ---------------------------
# Dataset Partition Mapping
# ---------------------------
partition_map = {
    0: "1_CancerGov_QA/qa_data.json",
    1: "2_GARD_QA/qa_data.json",
    2: "3_GHR_QA/qa_data.json",
    3: "4_MPlus_Health_Topics_QA/qa_data.json",
    4: "5_NIDDK_QA/qa_data.json",
    5: "6_NINDS_QA/qa_data.json",
    6: "7_SeniorHealth_QA/qa_data.json",
    7: "8_NHLBI_QA_XML/qa_data.json",
    8: "9_CDC_QA/qa_data.json",
}

dataset_root = "../../datasets"

def prepend_dataset_root(mapping):
    return {k: os.path.join(dataset_root, v) for k, v in mapping.items()}

partition_id_map = prepend_dataset_root(partition_map)

# ---------------------------
# Data Loading Function
# ---------------------------
def load_data(
    partition_id: int,
    model_name: str,
    batch_size: int = 16,
    chunk_size: int = 256,
    mlm_probability: float = 0.15,
    tokenize_columns=["answer"],
    remove_columns=["answer", "token_type_ids", "question"]
):
    """
    Loads and tokenizes dataset for masked language modeling (MLM).
    """
    if partition_id not in partition_id_map:
        raise ValueError(f"Invalid partition_id: {partition_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_path = partition_id_map[partition_id]
    logger.info(f"Loading dataset from: {data_path}")

    # Tokenize answers
    def tokenize_function(examples):
        texts = [" ".join(row) for row in zip(*[examples[col] for col in tokenize_columns])]
        return tokenizer(texts, truncation=False)

    # Chunk tokens into fixed-length blocks
    def chunk_function(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = len(concatenated["input_ids"])
        total_len = (total_len // chunk_size) * chunk_size

        result = {
            k: [t[i:i + chunk_size] for i in range(0, total_len, chunk_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Load, tokenize, and chunk the dataset
    try:
        dataset = load_dataset("json", data_files=[data_path])
        dataset = dataset.map(tokenize_function, batched=True)
        dataset = dataset.remove_columns(remove_columns)
        dataset = dataset.map(chunk_function, batched=True)

        if "train" not in dataset:
            raise ValueError(f"Expected 'train' split in dataset at {data_path}")
        
        dataset = dataset["train"].train_test_split(test_size=0.2)
    except Exception as e:
        logger.error(f"Failed to load or process dataset: {e}")
        raise

    # DataLoader preparation
    collate_fn = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    trainloader = DataLoader(
        dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    testloader = DataLoader(
        dataset["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    logger.info(f"Partition {partition_id} loaded: {len(trainloader.dataset)} training samples, {len(testloader.dataset)} test samples")

    return trainloader, testloader
