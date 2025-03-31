"""fl-mlm: A Flower / HuggingFace client app."""

import sys
import torch
import traceback
from flwr.common import Context
from flwr.client import ClientApp, NumPyClient

# Append shared code directory
sys.path.append("../llm_common")

from dataloader import load_data
from utils import log_to_file, setup_logger
from model import (
    load_model_with_lora,
    get_lora_state_dict,
    set_lora_state_dict,
    train,
    test,
)

logger = setup_logger("flower_client")

# Aliases for consistency
load_model = load_model_with_lora
get_weights = get_lora_state_dict
set_weights = set_lora_state_dict

# ------------------------------
# Flower Client Class Definition
# ------------------------------
class FlowerClient(NumPyClient):
    def __init__(self, client_id, net, trainloader, testloader, local_epochs, initial_lr=5e-5):
        self.client_id = client_id
        self.net = net.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.initial_lr = initial_lr
        self.device = self.net.device
        logger.info(f"Initialized FlowerClient {client_id} on device: {self.device}")

    def fit(self, parameters, config):
        try:
            results_dir = config.get("results-dir", "results_default")
            round_number = config.get("server-round", 0)

            set_weights(self.net, parameters)

            if round_number == 1:
                loss, acc = test(self.net, self.testloader, self.device, 0)
                logger.info(f"[Client {self.client_id}] Pre-train eval - Loss: {loss:.4f}, Acc: {acc:.4f}")
                log_to_file(
                    {"server_round": 0, "loss": loss, "accuracy": acc},
                    f"{results_dir}/client_before_{self.client_id}.jsonl"
                )

            train(self.net, self.trainloader, self.local_epochs, self.device, round_number, self.initial_lr)

            loss, acc = test(self.net, self.testloader, self.device, round_number)
            logger.info(f"[Client {self.client_id}] Post-train eval - Loss: {loss:.4f}, Acc: {acc:.4f}")

            log_to_file(
                {"server_round": round_number, "loss": loss, "accuracy": acc},
                f"{results_dir}/client_before_{self.client_id}.jsonl"
            )
        except Exception as e:
            logger.exception(f"[Client {self.client_id}] Exception during fit: {e}")
            raise

        return get_weights(self.net), len(self.trainloader), {"loss": loss, "accuracy": acc}

    def evaluate(self, parameters, config):
        try:
            set_weights(self.net, parameters)
            round_number = config.get("server-round", 0)
            results_dir = config.get("results-dir", "results_default")

            loss, acc = test(self.net, self.testloader, self.device, round_number)
            logger.info(f"[Client {self.client_id}] Aggregated eval - Loss: {loss:.4f}, Acc: {acc:.4f}")

            log_to_file(
                {"server_round": round_number, "loss": loss, "accuracy": acc},
                f"{results_dir}/client_after_{self.client_id}.jsonl"
            )
        except Exception as e:
            logger.exception(f"[Client {self.client_id}] Exception during evaluation: {e}")
            raise

        return float(loss), len(self.testloader), {"loss": loss, "accuracy": acc}

# ------------------------------
# Client Initialization Function
# ------------------------------
def client_fn(context: Context):
    # Extract runtime configs
    partition_id = context.node_config["partition-id"]
    cfg = context.run_config
    model_name = cfg["model-name"]

    # Load dataset partition
    trainloader, testloader = load_data(
        partition_id=partition_id,
        model_name=model_name,
        batch_size=cfg.get("batch-size", 16),
        chunk_size=cfg.get("chunk-size", 256),
        mlm_probability=cfg.get("mlm-probability", 0.15),
    )

    # Load model with LoRA params
    net = load_model(
        model_name=model_name,
        lora_alpha=cfg["lora-alpha"],
        lora_rank=cfg["lora-rank"],
        lora_dropout=cfg["lora-dropout"],
    )

    # Training config
    local_epochs = cfg["local-epochs"]
    initial_lr = cfg.get("initial-lr", 5e-5)

    # Return Flower client
    return FlowerClient(partition_id, net, trainloader, testloader, local_epochs, initial_lr).to_client()

# ------------------------------
# Flower Client Application
# ------------------------------
app = ClientApp(client_fn)
