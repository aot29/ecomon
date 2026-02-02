import os
import time
from backend.worker.tasks.utils.site_tasks import wait_for_lock_and_create_report
import soundfile as sf
from pathlib import Path
from datetime import datetime
from celery.utils.log import get_task_logger


from backend.worker.app import app
from backend.shared.models.db.models import Records, SiteDirectories
from backend.worker.tools import parse_datetime
from backend.worker.settings import WorkerSettings
from backend.worker.services.job_service import JobService
from backend.worker.database import db_session
from backend.worker.tasks.base_task import BaseTask

logger = get_task_logger(__name__)
settings = WorkerSettings()

# Configure logger level from settings
logger.setLevel(settings.log_level)


@app.task(
    name="delete_records_from_site",
    bind=True,
    base=BaseTask,
    track_started=True,
    queue="db_worker_queue",
)
def delete_records_from_site_task(self, site_id: int, directories: list[str]):
    job_id = self.request.id
    session = db_session()

    logger.info(f"Deleting records from site {site_id} in directories {directories}")

    try:
        deleted_records = 0
        counter = 0
        failed_directories = []
        skipped_directories = []
        
        for directory in directories:
            if self.check_revoked():
                return {
                    "status": "revoked",
                    "task_id": job_id,
                    "message": "Task was revoked.",
                }

            try:
                # Direct filtered delete
                deleted_count = (
                    session.query(Records)
                    .filter(
                        Records.site_id == site_id,
                        Records.filepath.like(f"{directory}%"),
                    )
                    .delete(synchronize_session=False)
                )

                session.commit()
                deleted_records += deleted_count
                counter += 1
                logger.info(f"Deleted {deleted_count} records from {directory}")
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to delete records from directory {directory}: {str(e)}")
                failed_directories.append({"directory": directory, "error": str(e)})
                skipped_directories.append(directory)
                counter += 1
                # Continue with next directory
            
            # Progress updates with separate short-lived session
            try:
                JobService.update_job_progress_by_counter(
                    session, job_id, counter, len(directories)
                )
                session.commit()
                time.sleep(1)
            except Exception as e:
                session.rollback()
                logger.error(f"Progress update failed: {str(e)}")
            
            try:
                JobService.updateResult(
                    session,
                    job_id,
                    {
                        "deleted_records": deleted_records,
                        "failed_directories": failed_directories,
                        "skipped_count": len(skipped_directories),
                    },
                )
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to update job result: {str(e)}")
        # Log summary of skipped directories if any
        if skipped_directories:
            logger.warning(
                f"Skipped {len(skipped_directories)} directories due to errors: {', '.join(skipped_directories)}"
            )
        
        if deleted_records == 0:
            message = "No records found to delete"
            if failed_directories:
                message += f". {len(failed_directories)} directories failed."
            return {
                "status": "success",
                "message": message,
                "failed_directories": failed_directories,
            }
        wait_for_lock_and_create_report(job_id, site_id, session, logger)
    except Exception as e:
        session.rollback()
        JobService.set_job_error(session, job_id, str(e))
        logger.error(f"Error deleting records: {str(e)}")
        raise e

    return {
        "status": "success",
        "message": f"Successfully deleted {deleted_records} records from {len(directories)} directories",
    }
