import os
import subprocess
import shutil
import time
import pandas
import numpy as np
from io import StringIO
from sqlalchemy import func, text, Text
from datetime import datetime
from celery.utils.log import get_task_logger
from collections import namedtuple
from backend.worker.app import app
from backend.shared.models.db.models import (
    ModelInferenceLogs,
    Models,
    Records,
    ModelInferenceResults,
)

from backend.worker.tools import parse_datetime
from backend.worker.settings import WorkerSettings
from backend.worker.services.job_service import JobService

from backend.worker.database import db_session
from backend.worker.tasks.base_task import BaseTask

from backend.shared.consts import task_topic

logger = get_task_logger(__name__)
settings = WorkerSettings()

# Configure logger level from settings
logger.setLevel(settings.log_level)

BATCH_SIZE = 100  # Reduced from 1000 to lower memory usage

def setup_database_optimizations(session):
    """Configure database session for optimal bulk operations."""
    logger.info("Optimizing database session for bulk inserts")
    session.execute(text("SET session_replication_role = replica"))  # Skip FK triggers
    session.execute(text("SET work_mem = '512MB'"))
    session.execute(text("SET maintenance_work_mem = '1GB'"))
    session.execute(text("SET synchronous_commit = OFF"))
    session.execute(text("SET commit_delay = 100000"))
    session.execute(text("SET commit_siblings = 5"))
    session.commit()

def create_temp_directories(job_id):
    """Create temporary directories for job processing."""
    job_temp_dir = os.path.join(settings.tmp_dir, job_id)
    host_model_output_dir = os.path.join(settings.host_tmp_dir, job_id)
    input_paths_file = os.path.join(settings.tmp_dir, job_id, "inputPaths.txt")
    host_input_paths_file = os.path.join(settings.host_tmp_dir, job_id, "inputPaths.txt")

    os.makedirs(job_temp_dir, exist_ok=True)
    return job_temp_dir, host_model_output_dir, input_paths_file, host_input_paths_file

def get_model_data(session, model_id):
    """Retrieve model data from database."""
    model = (
        session.query(
            Models.name,
            Models.additional_docker_arguments,
            Models.additional_model_arguments,
            Models.image,
            Models.segment_duration
        )
        .filter(Models.id == model_id)
        .first()
    )

    if not model:
        raise Exception(f"Model {model_id} not found")

    # Add debug logging to verify
    logger.info(f"Raw query result - name: {model[0]}, image: {model[3]}, segment_duration: {model[4]}")

    # detach model from session
    ModelData = namedtuple(
        "ModelData",
        ["name", "additional_docker_arguments", "additional_model_arguments", "image", "segment_duration"],
    )
    return ModelData(*model)

def count_records_to_process(session, site_id, model_id, start_datetime, end_datetime):
    """Count records that need to be processed for the given parameters."""
    logger.info(f"Fetching records for site {site_id} and model {model_id}")
    total_count = (
        session.query(func.count(Records.id))
        .join(
            ModelInferenceLogs,
            (Records.id == ModelInferenceLogs.record_id)
            & (ModelInferenceLogs.model_id == model_id),
            isouter=True,
        )
        .filter(Records.site_id == site_id, ModelInferenceLogs.id.is_(None))
        .filter(Records.record_datetime >= start_datetime)
        .filter(Records.record_datetime <= end_datetime)
        .filter(
            (Records.errors.is_(None))
            | (Records.errors == text("'null'::jsonb"))
            | (Records.errors.cast(Text).like('%duration_mismatch%'))
        )  # Skip records with errors except duration_mismatch
        .scalar()
    )

    logger.info(f"Found {total_count} records to process for site {site_id}")
    return total_count

def get_batch_records(session, site_id, model_id, batch_size):
    """Get a batch of records to process."""
    records = (
        session.query(Records.id, Records.filepath, Records.filename)
        .outerjoin(
            ModelInferenceLogs,
            (Records.id == ModelInferenceLogs.record_id)
            & (ModelInferenceLogs.model_id == model_id),
        )
        .filter(Records.site_id == site_id)
        .filter(ModelInferenceLogs.id.is_(None))
        .filter(
            (Records.errors.is_(None))
            | (Records.errors == text("'null'::jsonb"))
            | (Records.errors.cast(Text).like('%duration_mismatch%'))
        )  # Skip records with errors except duration_mismatch
        .limit(batch_size)
        .all()
    )
    return records

def prepare_input_paths_file(records, input_paths_file, record_name_to_id):
    """Create input paths file and mapping from filename to record ID."""
    with open(input_paths_file, "w") as f:
        for record in records:
            f.write(
                os.path.join(settings.host_base_data_directory, record.filepath)
                + "\n"
            )
            record_name_to_id[record.filename] = record.id

def build_docker_command(model, host_input_paths_file, host_model_output_dir, job_id):
    """Construct the Docker command to run the model."""
    docker_volumes = [
        f"-v {host_input_paths_file}:/app/inputPaths.txt",
        f"-v {host_model_output_dir}:/output",
        f"-v {settings.host_base_data_directory}:/data",
    ]

    command_parts = [
        "docker run",
        "-v /var/run/docker.sock:/var/run/docker.sock",  # needed for docker in docker
        "--rm",  # remove the container after running
        *docker_volumes,
        *(
            [model.additional_docker_arguments]  # additional docker arguments
            if model.additional_docker_arguments
            else []
        ),
        "ghcr.io/mfn-berlin/birdid-model-zoo:latest",
        *(
            [model.additional_model_arguments]
            if model.additional_model_arguments
            else []
        ),  # additional models arguments
        f"-i /app/inputPaths.txt",
        f"-m {model.image}",
        f"-o /output",
        f"-ov {host_model_output_dir}",
        *(f"--segmentDuration {model.segment_duration}".split() if model.segment_duration else []),
        "--removeTemporaryResultFile",
        f"-chown {os.getuid()}:{os.getgid()}",
        "--f pkl",
        *(
            [f"--gpuIx {settings.use_gpu}"]
            if settings.use_gpu.lower() != "none"
            else []
        ),  # gpu
        f"-w {job_id}",
        "-on output",
    ]

    return " ".join(command_parts)

def run_docker_command(command):
    """Execute the Docker command and handle output."""
    logger.info(f"Running command: {command}")

    try:
        process = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        logger.info(f"Command output: {process.stdout}")
        if process.stderr:
            logger.warning(f"Command stderr: {process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Command stderr: {e.stderr}")
        raise Exception(f"Docker command failed: {e.stderr}")

def analyze_dataframe_structure(df):
    """Log detailed information about the dataframe structure."""
    logger.info("=== DATAFRAME STRUCTURE FROM PICKLE FILE ===")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Columns ({len(df.columns)}): {df.columns.tolist()}")

    # Print column details
    logger.info("\nColumn details:")
    for col in df.columns:
        logger.info(f"- {col}: {df[col].dtype} (unique values: {df[col].nunique()}, "
                  f"NaN count: {df[col].isna().sum()})")
    logger.info("=== END OF DATAFRAME STRUCTURE ===")

    # Print first 10 rows
    logger.info("\nFirst 10 rows:")
    logger.info("\n" + df.head(10).to_string())
    logger.info("=== END OF DATAFRAME PREVIEW ===")

def log_problematic_rows(df):
    """Identify and log problematic rows in the dataframe."""
    # Find all rows with problematic label_id values
    problematic_rows = df[df['label_id'].isna() | df['label_id'].isin([np.inf, -np.inf]) |
                        df['label_id'].apply(lambda x: isinstance(x, float) and not x.is_integer())]

    if not problematic_rows.empty:
        logger.warning(f"\nFound {len(problematic_rows)} problematic rows:")
        logger.warning("=== PROBLEMATIC ROWS DETAILS ===")
        logger.warning(f"{'Index':<6} {'Model':<6} {'Filename':<20} {'Start':<6} {'End':<6} "
                      f"{'Conf':<6} {'Label_ID':<10} {'Label_Model'}")
        logger.warning("-" * 80)

        for idx, row in problematic_rows.iterrows():
            logger.warning(f"{idx:<6} {row.get('model_id', 'N/A'):<6} "
                          f"{row.get('filename', 'N/A')[:20]:<20} "
                          f"{row.get('start_time', 'N/A'):<6.2f} "
                          f"{row.get('end_time', 'N/A'):<6.2f} "
                          f"{row.get('confidence', 'N/A'):<6.2f} "
                          f"{str(row.get('label_id', 'N/A')):<10} "
                          f"{str(row.get('label_model', 'N/A'))}")
        logger.warning("=== END OF PROBLEMATIC ROWS ===")
    else:
        logger.info("No problematic rows found in original DataFrame")

def clean_and_prepare_results(df, record_name_to_id, model_id):
    """Clean and prepare the results dataframe for database insertion."""
    # Filter by confidence
    df = df[df["confidence"] >= 0.01]
    if len(df) == 0:
        return None

    # Map filename to record_id and add model_id
    df["record_id"] = df["filename"].map(record_name_to_id)
    df["model_id"] = model_id

    # Sort by record_id for partition efficiency
    df = df.sort_values("record_id")

    # Select and reorder columns for insertion
    df_results = df[["record_id", "model_id", "start_time", "end_time", "confidence", "label_id"]]

    # Log and handle problematic values
    logger.info(f"DataFrame shape before cleaning: {df_results.shape}")
    logger.info(f"label_id column dtype: {df_results['label_id'].dtype}")
    logger.info(f"label_id value counts:\n{df_results['label_id'].value_counts(dropna=False).head(20)}")

    # Check for specific problematic values
    problematic = df_results[~df_results['label_id'].apply(
        lambda x: isinstance(x, (int, float)) and not (np.isnan(x) or np.isinf(x)) or x is None
    )]
    if not problematic.empty:
        logger.warning(f"Found {len(problematic)} problematic label_id values")
        logger.debug(f"Problematic records:\n{problematic[['record_id', 'label_id']].to_string()}")

    # Handle NaN/inf values in label_id
    original_count = len(df_results)
    df_results = df_results.dropna(subset=["label_id"])  # Remove rows with NaN label_id
    df_results = df_results[~df_results["label_id"].isin([np.inf, -np.inf])]  # Remove rows with inf label_id

    if len(df_results) < original_count:
        logger.warning(f"Dropped {original_count - len(df_results)} records with invalid label_id values")

    # Make sure label_id is integer
    df_results["label_id"] = df_results["label_id"].astype(int)

    return df_results

def insert_results_with_copy(session, df_results, model):
    """Insert results into database using COPY for maximum performance."""
    if not model or not model.name:
        error_msg = f"Model name not found. Cannot determine target table."
        logger.error(error_msg)
        raise Exception(error_msg)

    table_name = "model_inference_results_pt_record"
    logger.info(f"Writing to table: {table_name}")
    logger.info(f"Inserting {len(df_results)} results using COPY")

    # Create CSV buffer in memory
    buffer = StringIO()
    df_results.to_csv(buffer, index=False, header=False, sep='\t', na_rep='\\N')
    buffer.seek(0)

    # COPY from buffer to table
    connection = session.connection().connection
    cursor = connection.cursor()
    cursor.copy_expert(
        f"""
        COPY "{table_name}"
        (record_id, model_id, start_time, end_time, confidence, label_id)
        FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '\\N')
        """,
        buffer
    )

    return cursor

def insert_logs_with_copy(session, model_id, record_ids):
    """Insert inference logs using COPY for performance."""
    logs_df = pandas.DataFrame({
        "model_id": model_id,
        "record_id": sorted(record_ids),
        "analyzed": True
    })

    logger.info(f"Inserting {len(logs_df)} logs using COPY")
    logs_buffer = StringIO()
    logs_df.to_csv(logs_buffer, index=False, header=False, sep='\t')
    logs_buffer.seek(0)

    connection = session.connection().connection
    cursor = connection.cursor()
    cursor.copy_expert(
        """
        COPY model_inference_logs
        (model_id, record_id, analyzed)
        FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t')
        """,
        logs_buffer
    )

    return cursor

def cleanup_temp_files(job_temp_dir):
    """Remove temporary files after processing."""
    for file in os.listdir(job_temp_dir):
        os.remove(os.path.join(job_temp_dir, file))

@app.task(
    name=f"{task_topic.MODEL_INFERENCE_SITE.value}",
    bind=True,
    base=BaseTask,
    track_started=True,
    queue="inference_queue",
)
def model_inference_site_task(
    self, site_id: int, model_id: int, start_datetime: datetime, end_datetime: datetime
):
    job_id = self.request.id
    session = db_session()
    file_counter = 0

    # Track skipped records due to batch errors
    skipped_batches = []
    total_skipped_records = 0

    # create temp directories
    job_temp_dir, host_model_output_dir, input_paths_file, host_input_paths_file = create_temp_directories(job_id)
    workerId = job_id

    try:
        # Get model data
        model = get_model_data(session, model_id)

        # Configure database for bulk operations
        setup_database_optimizations(session)

        # Count records to process
        total_count = count_records_to_process(session, site_id, model_id, start_datetime, end_datetime)

        if total_count == 0:
            JobService.update_job_progress(session, job_id, 100)
            return {
                "status": "success",
                "message": f"No records to process for site {site_id}",
            }

        while total_count > file_counter:
            session.autoflush = False
            if self.check_revoked():
                time.sleep(1)
                return {
                    "status": "revoked",
                    "message": "Task was revoked.",
                }

            # Get batch of records
            records = get_batch_records(session, site_id, model_id, BATCH_SIZE)
            if len(records) == 0:
                break

            # Store record IDs in this batch for error tracking
            batch_record_ids = [record.id for record in records]
            batch_filenames = [record.filename for record in records]

            try:
                # Prepare input paths file
                record_name_to_id = {}
                prepare_input_paths_file(records, input_paths_file, record_name_to_id)

                # Build and run Docker command
                command = build_docker_command(model, host_input_paths_file, host_model_output_dir, workerId)
                run_docker_command(command)

                # Read and process results
                df = pandas.read_pickle(os.path.join(job_temp_dir, "output.pkl"))

                # Analyze dataframe structure
                analyze_dataframe_structure(df)
                log_problematic_rows(df)

                # Clean and prepare results
                df_results = clean_and_prepare_results(df, record_name_to_id, model_id)
                if df_results is None:
                    logger.warning(f"No valid results after filtering for batch of {len(records)} records")
                    # Still count as processed since logs will be created
                    file_counter += len(records)
                    # Insert logs even when no results (records were processed, just no detections)
                    logs_cursor = insert_logs_with_copy(session, model_id, batch_record_ids)
                    logs_cursor.close()
                    session.commit()
                    continue

                # Insert results and logs
                results_cursor = insert_results_with_copy(session, df_results, model)
                logs_cursor = insert_logs_with_copy(session, model_id, df["record_id"].unique())

                # Clean up
                results_cursor.close()
                logs_cursor.close()
                session.commit()
                session.close()
                db_session.remove()
                session = db_session()

                # Re-apply session optimizations
                setup_database_optimizations(session)

                file_counter += len(records)

            except Exception as batch_error:
                # Log the batch error and track skipped records
                logger.error(f"Batch processing failed for {len(records)} records: {str(batch_error)}")
                logger.error(f"Failed batch record IDs: {batch_record_ids[:10]}{'...' if len(batch_record_ids) > 10 else ''}")
                logger.error(f"Failed batch filenames: {batch_filenames[:10]}{'...' if len(batch_filenames) > 10 else ''}")

                skipped_batches.append({
                    "record_count": len(records),
                    "record_ids": batch_record_ids[:20],  # Store first 20 IDs
                    "filenames": batch_filenames[:20],    # Store first 20 filenames
                    "error": str(batch_error)
                })
                total_skipped_records += len(records)

                # Clean up any partial files
                try:
                    cleanup_temp_files(job_temp_dir)
                except:
                    pass

                # Continue to next batch instead of failing entire task
                continue

            JobService.update_job_progress_by_counter(
                session, job_id, file_counter, total_count
            )

            cleanup_temp_files(job_temp_dir)
            JobService.updateResult(session, job_id, {
                "inferred_records": file_counter,
                "skipped_records": total_skipped_records,
                "skipped_batches_count": len(skipped_batches)
            })

        JobService.update_job_progress(session, job_id, 100)

        # Log summary of skipped records
        if total_skipped_records > 0:
            logger.warning("=" * 80)
            logger.warning("BATCH PROCESSING ERRORS SUMMARY")
            logger.warning("=" * 80)
            logger.warning(f"Total records skipped due to batch errors: {total_skipped_records}")
            logger.warning(f"Number of failed batches: {len(skipped_batches)}")

            for idx, batch_info in enumerate(skipped_batches, 1):
                logger.warning(f"\nFailed Batch #{idx}:")
                logger.warning(f"  Records in batch: {batch_info['record_count']}")
                logger.warning(f"  Error: {batch_info['error']}")
                logger.warning(f"  Sample record IDs: {batch_info['record_ids'][:5]}")
                logger.warning(f"  Sample filenames: {batch_info['filenames'][:5]}")

            logger.warning("=" * 80)

            # Update final result with detailed error info
            JobService.updateResult(session, job_id, {
                "inferred_records": file_counter,
                "skipped_records": total_skipped_records,
                "failed_batches": skipped_batches
            })

        logger.info(f"Task completed: {file_counter} records processed successfully, {total_skipped_records} records skipped due to errors")

    except Exception as e:
        session.rollback()
        JobService.set_job_error(session, job_id, str(e))
        logger.error(f"Task failed: {str(e)}")
        raise e
    finally:
        # Re-enable normal operation
        try:
            session.execute(text("SET session_replication_role = DEFAULT"))
            session.commit()
        except:
            pass
        # Clean up temp directory
        if os.path.exists(job_temp_dir):
            shutil.rmtree(job_temp_dir)

    return {
        "status": "success",
        "message": f"Successfully analyzed {file_counter} records for site {site_id}",
    }