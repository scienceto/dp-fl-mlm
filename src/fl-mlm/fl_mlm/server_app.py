"""fl-mlm: A Flower / HuggingFace app."""

import sys
import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

# Add LLM common utilities
sys.path.append("../llm_common")
from model import (
    load_model_with_lora,
    get_lora_state_dict,
    set_lora_state_dict,
    test,
)
from utils import create_results_dir, log_to_file, setup_logger
from dataloader import load_data

logger = setup_logger("flower_server")

# Aliases for Flower
get_weights = get_lora_state_dict
set_weights = set_lora_state_dict
load_model = load_model_with_lora

# Utility to initialize the model with LoRA
def initialize_model(context):
    model_name = context.run_config["model-name"]
    logger.info(f"Initializing model '{model_name}' with LoRA.")
    return load_model(
        model_name=model_name,
        lora_alpha=context.run_config["lora-alpha"],
        lora_rank=context.run_config["lora-rank"],
        lora_dropout=context.run_config["lora-dropout"],
    )

# Optional evaluation function if test dataset is provided
def build_evaluate_fn(context, results_dir, model_name, device):
    if context.run_config["test-dataset"] < 0:
        logger.info("No test dataset provided. Skipping evaluation function.")
        return None

    logger.info(f"Loading test dataset for partition ID: {context.run_config['test-dataset']}")
    _, testloader = load_data(
        partition_id=context.run_config["test-dataset"],
        model_name=model_name,
        batch_size=context.run_config.get("batch-size", 16),
        chunk_size=context.run_config.get("chunk-size", 256),
        mlm_probability=context.run_config.get("mlm-probability", 0.15),
    )

    def evaluate_fn(server_round, parameters, config):
        logger.info(f"Evaluating model at round {server_round}")
        net = initialize_model(context)
        set_weights(net, parameters)
        net.to(device)

        loss, accuracy = test(net, testloader, device, server_round)
        log_to_file(
            {"server_round": server_round, "loss": loss, "accuracy": accuracy},
            f"{results_dir}/test_dataset.jsonl",
        )

        logger.info(f"Test Evaluation | Round {server_round} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")
        return loss, {"accuracy": accuracy}

    return evaluate_fn

# Configuration per round
def config_fn_factory(results_dir):
    def config_fn(server_round):
        return {
            "server-round": server_round,
            "results-dir": results_dir,
        }
    return config_fn

# Main server function for Flower
def server_fn(context: Context):
    logger.info("Starting Flower server...")

    results_dir = create_results_dir()
    logger.info(f"Results will be stored in: {results_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = context.run_config["model-name"]

    # Initialize global model and weights
    net = initialize_model(context)
    initial_parameters = ndarrays_to_parameters(get_weights(net))
    logger.info("Initial model weights extracted.")

    # Evaluation function (optional)
    evaluate_fn = build_evaluate_fn(context, results_dir, model_name, device)

    # Strategy config
    strategy = FedAvg(
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=context.run_config["fraction-evaluate"],
        initial_parameters=initial_parameters,
        on_fit_config_fn=config_fn_factory(results_dir),
        on_evaluate_config_fn=config_fn_factory(results_dir),
        evaluate_fn=evaluate_fn,
        accept_failures=False,
    )

    # Server config
    config = ServerConfig(num_rounds=context.run_config["num-server-rounds"])
    logger.info("FedAvg strategy and server configuration complete.")

    return ServerAppComponents(strategy=strategy, config=config)

# Flower App entrypoint
app = ServerApp(server_fn=server_fn)