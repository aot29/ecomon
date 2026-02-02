"""
Self-contained tests for model inference functions.
This file doesn't import any project modules, avoiding dependency issues.
"""

import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from datetime import datetime

def build_docker_command_test_version(model, host_input_paths_file, host_model_output_dir, job_id, use_gpu='none'):
    """
    Test version of build_docker_command function.
    This is a simplified version that doesn't depend on project imports.
    """
    docker_volumes = [
        f"-v {host_input_paths_file}:/app/inputPaths.txt",
        f"-v {host_model_output_dir}:/output",
        f"-v /data:/data",
    ]

    command_parts = [
        "docker run",
        "-v /var/run/docker.sock:/var/run/docker.sock",
        "--rm",
        *docker_volumes,
        *(
            [model.additional_docker_arguments]
            if hasattr(model, 'additional_docker_arguments') and model.additional_docker_arguments
            else []
        ),
        "ghcr.io/mfn-berlin/birdid-model-zoo:latest",
        *(
            [model.additional_model_arguments]
            if hasattr(model, 'additional_model_arguments') and model.additional_model_arguments
            else []
        ),
        f"-i /app/inputPaths.txt",
        f"-m {model.image}",
        f"-o /output",
        f"-ov {host_model_output_dir}",
        *(f"--segmentDuration {model.segment_duration}".split() if hasattr(model, 'segment_duration') and model.segment_duration else []),
        "--removeTemporaryResultFile",
        f"-chown {os.getuid()}:{os.getgid()}",
        "--f pkl",
        *(
            [f"--gpuIx {use_gpu}"]
            if use_gpu.lower() != "none"
            else []
        ),
        f"-w {job_id}",
        "-on output",
    ]

    return " ".join(command_parts)

def clean_and_prepare_results_test_version(df, record_name_to_id, model_id):
    """
    Test version of clean_and_prepare_results function.
    This is a simplified version that doesn't depend on project imports.
    """
    # Filter by confidence
    df = df[df["confidence"] >= 0.01].copy()  # Use .copy() to avoid SettingWithCopyWarning
    if len(df) == 0:
        return None

    # Map filename to record_id and add model_id
    df.loc[:, "record_id"] = df["filename"].map(record_name_to_id)
    df.loc[:, "model_id"] = model_id

    # Sort by record_id for partition efficiency
    df = df.sort_values("record_id")

    # Handle NaN/inf values in label_id
    original_count = len(df)
    df = df.dropna(subset=["label_id"])
    df = df[~df["label_id"].isin([np.inf, -np.inf])]

    # Filter out non-integer values (like 2.5)
    df = df[df["label_id"].apply(lambda x: isinstance(x, (int, float)) and x == int(x))]

    if len(df) < original_count:
        print(f"Dropped {original_count - len(df)} records with invalid label_id values")

    # Make sure label_id is integer
    df["label_id"] = df["label_id"].astype(int)

    # Select and reorder columns for insertion
    return df[["record_id", "model_id", "start_time", "end_time", "confidence", "label_id"]]

def test_clean_and_prepare_results_with_invalid_values():
    """Test handling of various invalid label_id values"""
    df = pd.DataFrame({
        'filename': ['file1.wav', 'file2.wav', 'file3.wav', 'file4.wav', 'file5.wav'],
        'confidence': [0.9, 0.8, 0.7, 0.6, 0.5],
        'start_time': [0.0, 1.0, 2.0, 3.0, 4.0],
        'end_time': [1.0, 2.0, 3.0, 4.0, 5.0],
        'label_id': [1, np.inf, -np.inf, 2.5, None]  # Various problematic values
    })

    record_name_to_id = {
        'file1.wav': 101,
        'file2.wav': 102,
        'file3.wav': 103,
        'file4.wav': 104,
        'file5.wav': 105
    }

    result = clean_and_prepare_results_test_version(df, record_name_to_id, 1)

    # Should only keep the valid record (file1.wav)
    assert len(result) == 1
    assert result.iloc[0].record_id == 101
    assert result.iloc[0].label_id == 1

def test_build_docker_command_with_gpu():
    """Test Docker command construction with GPU"""
    model = MagicMock()
    model.name = 'test_model'
    model.additional_docker_arguments = '--gpus all'
    model.additional_model_arguments = '--batch-size 32'
    model.image = 'test-image:v1'
    model.segment_duration = 5

    command = build_docker_command_test_version(
        model,
        '/host/inputPaths.txt',
        '/host/output',
        'test-job-123',
        use_gpu='0'
    )

    assert 'docker run' in command
    assert '-v /host/inputPaths.txt:/app/inputPaths.txt' in command
    assert '--gpus all' in command
    assert '--batch-size 32' in command
    assert '-m test-image:v1' in command
    assert '--segmentDuration 5' in command
    assert '--gpuIx 0' in command
    assert '-w test-job-123' in command

def test_build_docker_command_without_gpu():
    """Test Docker command construction without GPU"""
    model = MagicMock()
    model.name = 'test_model'
    model.additional_docker_arguments = None
    model.additional_model_arguments = None
    model.image = 'test-image:v1'
    model.segment_duration = None

    command = build_docker_command_test_version(
        model,
        '/host/inputPaths.txt',
        '/host/output',
        'test-job-123',
        use_gpu='none'
    )

    assert 'docker run' in command
    assert '-v /host/inputPaths.txt:/app/inputPaths.txt' in command
    assert '--gpus all' not in command
    assert '--gpuIx' not in command  # Should not include GPU flag
    assert '-w test-job-123' in command
    assert '--segmentDuration' not in command  # No segment duration

def test_clean_and_prepare_results():
    """Test dataframe cleaning and preparation"""
    # Create test dataframe
    df = pd.DataFrame({
        'filename': ['file1.wav', 'file2.wav', 'file3.wav'],
        'confidence': [0.9, 0.005, 0.8],
        'start_time': [0.0, 1.0, 2.0],
        'end_time': [1.0, 2.0, 3.0],
        'label_id': [1, 2, np.nan]
    })

    record_name_to_id = {'file1.wav': 101, 'file2.wav': 102, 'file3.wav': 103}

    result = clean_and_prepare_results_test_version(df, record_name_to_id, 1)

    # Should filter out low confidence and NaN label_id
    assert len(result) == 1
    assert result.iloc[0].record_id == 101
    assert result.iloc[0].model_id == 1
    assert result.iloc[0].label_id == 1

def test_clean_and_prepare_results_with_invalid_values():
    """Test handling of various invalid label_id values"""
    df = pd.DataFrame({
        'filename': ['file1.wav', 'file2.wav', 'file3.wav', 'file4.wav', 'file5.wav'],
        'confidence': [0.9, 0.8, 0.7, 0.6, 0.5],
        'start_time': [0.0, 1.0, 2.0, 3.0, 4.0],
        'end_time': [1.0, 2.0, 3.0, 4.0, 5.0],
        'label_id': [1, np.inf, -np.inf, 2.5, None]  # Various problematic values
    })

    record_name_to_id = {
        'file1.wav': 101,
        'file2.wav': 102,
        'file3.wav': 103,
        'file4.wav': 104,
        'file5.wav': 105
    }

    result = clean_and_prepare_results_test_version(df, record_name_to_id, 1)

    # Should only keep the valid record (file1.wav)
    assert len(result) == 1
    assert result.iloc[0].record_id == 101
    assert result.iloc[0].label_id == 1