"""Utility functions for the Amazon deforestation prediction project."""

import argparse
import logging
import socket
from datetime import datetime

import torch
import yaml

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


# device methods
def get_device(device_str: str = "cpu") -> torch.device:
    """
    Returns a torch.device object based on the input string.
    If device_str is 'cuda' and CUDA is not available, falls back to CPU.
    """
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# argparse methods
def parse_args(args=None) -> argparse.Namespace:
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep Learning Project")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    return parser.parse_args(args)


# config methods
def load_config(path: str) -> dict:
    """
    Loads a configuration file from the given path.
    """
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# memory record methods
def start_record_memory_history(
    logger: logging.Logger, max_entries: int
) -> None:
    """Start recording CUDA memory history for debugging purposes.

    Args:
        logger: Logger instance for recording status messages
        max_entries: Maximum number of memory events to record

    Note:
        This function only works when CUDA is available. If CUDA is not
        available, it logs a message and returns without taking action.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(
        max_entries=max_entries
    )


def stop_record_memory_history(logger: logging.Logger) -> None:
    """Stop recording CUDA memory history.

    Args:
        logger: Logger instance for recording status messages

    Note:
        This function only works when CUDA is available. If CUDA is not
        available, it logs a message and returns without taking action.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)


def export_memory_snapshot(logger: logging.Logger) -> None:
    """Export CUDA memory snapshot to a pickle file for analysis.

    Args:
        logger: Logger instance for recording status messages

    Note:
        This function only works when CUDA is available. If CUDA is not
        available, it logs a message and returns without taking action.
        The actual filename will be in format: {hostname}_{timestamp}.pickle
    """
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not exporting memory snapshot")
        return

    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    try:
        logger.info("Saving snapshot to local file: %s.pickle", file_prefix)
        torch.cuda.memory._dump_snapshot(
            f"memory_records/{file_prefix}.pickle"
        )
    except (OSError, IOError) as e:
        logger.error(
            "Failed to save memory snapshot due to file system error: %s", e
        )
    except RuntimeError as e:
        logger.error(
            "Failed to capture memory snapshot due to CUDA runtime error: %s",
            e,
        )


# def trace_handler(prof: torch.profiler.profile):
#     """Export PyTorch profiler trace and memory timeline files.

#     Args:
#         prof: PyTorch profiler instance containing profiling data

#     Note:
#         This function exports two files:
#         - Chrome trace file (.json.gz) for performance analysis
#         - Memory timeline file (.html) for CUDA memory analysis
#         Both files are prefixed with hostname and timestamp.
#     """
#     # Prefix for file names.
#     host_name = socket.gethostname()
#     timestamp = datetime.now().strftime(TIME_FORMAT_STR)
#     file_prefix = f"{host_name}_{timestamp}"

#     # Construct the trace file.
#     prof.export_chrome_trace(f"{file_prefix}.json.gz")

#     # Construct the memory timeline file.
#     prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
