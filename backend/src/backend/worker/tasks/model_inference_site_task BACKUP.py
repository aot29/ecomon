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

# BATCH_SIZE = 1000  # Increased from 100 for better throughput
BATCH_SIZE = 100  # Reduced from 1000 to lower memory usage

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

    # Track skipped records (individual records, not batches)
    skipped_records = []
    failed_record_ids = set()

    # create temp directories for the pathes for host and container
    # If you start an container inside a container, you need to mount the host directory to the container
    # and use the host directory for the output and input
    job_temp_dir = os.path.join(settings.tmp_dir, job_id)
    host_model_output_dir = os.path.join(settings.host_tmp_dir, job_id)
    input_paths_file = os.path.join(settings.tmp_dir, job_id, "inputPaths.txt")
    host_input_paths_file = os.path.join(
        settings.host_tmp_dir, job_id, "inputPaths.txt"
    )

    # workerId will be passed to the name of the Docker container where the model runs.
    workerId = job_id

    try:
        # get model string
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

        # DEBUG: Log the raw query result
        logger.info(f"Raw query result - name: {model[0]}, image: {model[3]}, segment_duration: {model[4]}")

        # CRITICAL: Optimize session for bulk inserts on heavily indexed partitioned table
        logger.info("Optimizing database session for bulk inserts")
        session.execute(text("SET session_replication_role = replica"))  # Skip FK triggers
        session.execute(text("SET work_mem = '512MB'"))
        session.execute(text("SET maintenance_work_mem = '1GB'"))
        session.execute(text("SET synchronous_commit = OFF"))
        session.execute(text("SET commit_delay = 100000"))
        session.execute(text("SET commit_siblings = 5"))
        session.commit()

        # detach model from session
        ModelData = namedtuple(
            "ModelData",
            ["name", "additional_docker_arguments", "additional_model_arguments", "image", "segment_duration"],
        )
        model = ModelData(*model)

        # Add debug logging to verify
        logger.info(f"Model name: {model.name}")
        logger.info(f"Model image: {model.image}")
        logger.info(f"Model segment_duration: {model.segment_duration}")

        file_counter = 0
        # Get current user and group IDs to make the docker output files readable
        uid = os.getuid()
        gid = os.getgid()

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

        # create the tmp directory
        logger.info(
            f"Found {total_count} records to process for site {site_id} and model {model.name}"
        )
        os.makedirs(job_temp_dir, exist_ok=True)
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
                # Wait for 1 second to ensure the task is revoked
                return {
                    "status": "revoked",
                    "message": "Task was revoked.",
                }

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
                .limit(BATCH_SIZE)
                .all()
            )
            if len(records) == 0:
                break

            # Store record IDs in this batch for error tracking
            batch_record_ids = [record.id for record in records]
            batch_filenames = [record.filename for record in records]
            batch_filepaths = [record.filepath for record in records]

            # Track which records in this batch were successfully processed
            successfully_processed_ids = set()
            batch_failed_records = []

            try:
                # Prepare inputPaths.txt file for the model
                record_name_to_id = {}
            with open(input_paths_file, "w") as f:
                for record in records:
                    f.write(
                        os.path.join(settings.host_base_data_directory, record.filepath)
                        + "\n"
                    )
                    record_name_to_id[record.filename] = record.id
            docker_volumes = [
                f"-v {host_input_paths_file}:/app/inputPaths.txt",
                f"-v {host_model_output_dir}:/output",
                f"-v {settings.host_base_data_directory}:/data",
            ]
            # Build command as array for better readability and spacing control
            logger.info(f"settings.use_gpu: {settings.use_gpu}")
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
                # "model", # for local testing
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
                f"-chown {uid}:{gid}",
                "--f pkl",
                *(
                    [f"--gpuIx {settings.use_gpu}"]
                    if settings.use_gpu.lower() != "none"
                    else []
                ),  # gpu
                f"-w {workerId}",
                "-on output",
            ]

            command = " ".join(command_parts)

            logger.info(f"Running command: {command}")

            try:
                process = subprocess.run(
                    command, shell=True, check=True, capture_output=True, text=True
                )
                logger.info(f"Command output: {process.stdout}")
                if process.stderr:
                    logger.warning(f"Command stderr: {process.stderr}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with exit code {e.returncode}")
                logger.error(f"Command stderr: {e.stderr}")
                raise Exception(f"Docker command failed: {e.stderr}")

            # read the output.pkl file and add the results to the database
            df = pandas.read_pickle(os.path.join(job_temp_dir, "output.pkl"))

            # ====== ADD THIS DEBUGGING CODE RIGHT AFTER READING PICKLE ======
            # Print all column names and basic info
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

            # Identify and log all problematic rows before any filtering
            logger.info(f"Original DataFrame shape: {df.shape}")
            logger.info(f"Original DataFrame columns: {df.columns.tolist()}")

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
            # ====== END OF DEBUGGING CODE ======

            # if confidence is 0 or below confidence resolution
            df = df[df["confidence"] >= 0.01]
            if len(df) == 0:
                # No results to insert, skip to next batch
                continue

            # Map filename to record_id and add model_id
            df["record_id"] = df["filename"].map(record_name_to_id)
            df["model_id"] = model_id

            # CRITICAL: Sort by record_id for partition efficiency
            # This ensures inserts go to the same partition sequentially,
            # reducing partition switching overhead and improving cache utilization
            df = df.sort_values("record_id")

            # Select and reorder columns for insertion
            df_results = df[["record_id", "model_id", "start_time", "end_time", "confidence", "label_id"]]

            # ADD THIS DEBUGGING CODE RIGHT HERE:
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

            # Log and handle NaN/inf values in label_id before converting to integer
            nan_rows = df_results[df_results["label_id"].isna()]
            if not nan_rows.empty:
                logger.warning(f"Found {len(nan_rows)} records with NaN label_id values. Dropping these records.")
                logger.debug(f"NaN label_id records details: {nan_rows.to_dict()}")

            inf_rows = df_results[df_results["label_id"].isin([np.inf, -np.inf])]
            if not inf_rows.empty:
                logger.warning(f"Found {len(inf_rows)} records with infinite label_id values. Dropping these records.")
                logger.debug(f"Infinite label_id records details: {inf_rows.to_dict()}")

            # Handle NaN/inf values in label_id before converting to integer
            original_count = len(df_results)
            df_results = df_results.dropna(subset=["label_id"])  # Remove rows with NaN label_id
            df_results = df_results[~df_results["label_id"].isin([np.inf, -np.inf])]  # Remove rows with inf label_id

            if len(df_results) < original_count:
                logger.warning(f"Dropped {original_count - len(df_results)} records with invalid label_id values")

            # Make sure label_id is integer
            df_results["label_id"] = df_results["label_id"].astype(int)

            # Use COPY for maximum speed (10-50x faster than INSERT on indexed tables)
            logger.info(f"Inserting {len(df_results)} results using COPY")
            connection = session.connection().connection
            cursor = connection.cursor()

            # Construct table name with model name postfix
            if not model or not model.name:
                error_msg = f"Model name not found for model_id {model_id}. Cannot determine target table."
                logger.error(error_msg)
                raise Exception(error_msg)

            # table_name = f"model_inference_results_{model.name}"
            table_name = "model_inference_results_pt_record"
            logger.info(f"Writing to table: {table_name}")

            # Create CSV buffer in memory
            buffer = StringIO()
            df_results.to_csv(buffer, index=False, header=False, sep='\t', na_rep='\\N')
            buffer.seek(0)

            # COPY from buffer to table (bypasses most index overhead)
            # Use %s placeholder with quoted identifier to handle special characters
            cursor.copy_expert(
                f"""
                COPY "{table_name}"
                (record_id, model_id, start_time, end_time, confidence, label_id)
                FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '\\N')
                """,
                buffer
            )

            # Insert logs using COPY
            # Track which records were successfully processed in this batch
            successfully_processed_ids = set(df["record_id"].unique())

            logs_df = pandas.DataFrame({
                "model_id": model_id,
                "record_id": sorted(successfully_processed_ids),
                "analyzed": True
            })

            logger.info(f"Inserting {len(logs_df)} logs using COPY")
            logs_buffer = StringIO()
            logs_df.to_csv(logs_buffer, index=False, header=False, sep='\t')
            logs_buffer.seek(0)

            cursor.copy_expert(
                """
                COPY model_inference_logs
                (model_id, record_id, analyzed)
                FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t')
                """,
                logs_buffer
            )

            # Identify records in batch that weren't in the output (failed during Docker processing)
            all_batch_ids = set(batch_record_ids)
            missing_ids = all_batch_ids - successfully_processed_ids

            if missing_ids:
                logger.warning(f"Found {len(missing_ids)} records that failed during Docker processing")
                for record_id in missing_ids:
                    idx = batch_record_ids.index(record_id) if record_id in batch_record_ids else -1
                    if idx >= 0:
                        batch_failed_records.append({
                            "record_id": record_id,
                            "filename": batch_filenames[idx],
                            "filepath": batch_filepaths[idx],
                            "error": "Record not in Docker output (file read/processing error)"
                        })
                        failed_record_ids.add(record_id)

            cursor.close()
            session.commit()
            session.close()
            db_session.remove()
            session = db_session()

            # Re-apply session optimizations after reconnecting
            session.execute(text("SET session_replication_role = replica"))
            session.execute(text("SET work_mem = '512MB'"))
            session.execute(text("SET maintenance_work_mem = '1GB'"))
            session.execute(text("SET synchronous_commit = OFF"))
            session.execute(text("SET commit_delay = 100000"))
            session.execute(text("SET commit_siblings = 5"))
            session.commit()

            file_counter += len(records)

            # Add any failed records from this batch to the overall list
            if batch_failed_records:
                skipped_records.extend(batch_failed_records)
                logger.warning(f"Batch had {len(batch_failed_records)} failed records out of {len(records)} total")

            except Exception as batch_error:
                # Critical batch-level error (Docker, file system, database)
                logger.error(f"Critical batch error for {len(records)} records: {str(batch_error)}")
                logger.error(f"Error type: {type(batch_error).__name__}")

                # Mark all records in this batch as failed since we couldn't process any
                for idx, record_id in enumerate(batch_record_ids):
                    if record_id not in failed_record_ids:
                        skipped_records.append({
                            "record_id": record_id,
                            "filename": batch_filenames[idx],
                            "filepath": batch_filepaths[idx],
                            "error": f"Batch-level failure: {str(batch_error)}"
                        })
                        failed_record_ids.add(record_id)

                # Clean up any partial files
                try:
                    for file in os.listdir(job_temp_dir):
                        os.remove(os.path.join(job_temp_dir, file))
                except:
                    pass

                # Continue to next batch instead of failing entire task
                continue

            JobService.update_job_progress_by_counter(
                session, job_id, file_counter, total_count
            )
            # delete all files in the job_temp_dir for the next batch
            for file in os.listdir(job_temp_dir):
                os.remove(os.path.join(job_temp_dir, file))
            JobService.updateResult(session, job_id, {
                "inferred_records": file_counter,
                "skipped_records_count": len(skipped_records)
            })

        JobService.update_job_progress(session, job_id, 100)

        # Log summary of skipped records
        if skipped_records:
            logger.warning("=" * 80)
            logger.warning("SKIPPED RECORDS SUMMARY")
            logger.warning("=" * 80)
            logger.warning(f"Total records skipped: {len(skipped_records)}")

            # Group by error type for summary
            error_groups = {}
            for rec in skipped_records:
                error_key = rec["error"][:100]  # First 100 chars of error
                if error_key not in error_groups:
                    error_groups[error_key] = []
                error_groups[error_key].append(rec)

            logger.warning(f"\nErrors grouped by type ({len(error_groups)} unique errors):")
            for error_msg, records_list in sorted(error_groups.items(), key=lambda x: len(x[1]), reverse=True):
                logger.warning(f"\n  Error: {error_msg}")
                logger.warning(f"  Affected records: {len(records_list)}")
                logger.warning(f"  Sample record IDs: {[r['record_id'] for r in records_list[:5]]}")
                logger.warning(f"  Sample filenames: {[r['filename'] for r in records_list[:3]]}")

            logger.warning("=" * 80)

            # Update final result with detailed error info
            JobService.updateResult(session, job_id, {
                "inferred_records": file_counter,
                "skipped_records_count": len(skipped_records),
                "skipped_records": skipped_records[:100],  # Store first 100 for review
                "error_summary": {
                    error[:100]: len(recs) for error, recs in error_groups.items()
                }
            })
        else:
            logger.info("All records processed successfully, no records skipped")

        logger.info(f"Task completed: {file_counter} records processed, {len(skipped_records)} records skipped")

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
        # Clean up temp directory only if it exists
        if os.path.exists(job_temp_dir):
            shutil.rmtree(job_temp_dir)

    return {
        "status": "success",
        "message": f"Successfully analyzed {file_counter} records for site {site_id}",
    }